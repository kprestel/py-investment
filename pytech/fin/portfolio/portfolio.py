import logging
import queue
from abc import (
    ABCMeta,
    abstractmethod,
)
from datetime import datetime
from typing import (
    Dict,
    Iterable,
    List,
    TYPE_CHECKING,
)

import pandas as pd

import pytech.utils as utils
from pytech.backtest.event import SignalEvent
from pytech.data.handler import DataHandler
from pytech.fin.asset.owned_asset import OwnedAsset
from pytech.mongo import (
    ARCTIC_STORE,
    PortfolioStore,
)

if TYPE_CHECKING:
    from pytech.trading import (
    AnyOrder,
    Blotter,
)
from pytech.trading.trade import Trade
from pytech.utils.enums import (
    EventType,
    Position,
    TradeAction,
)
from pytech.utils.exceptions import (
    InsufficientFundsError,
    InvalidEventTypeError,
)

logger = logging.getLogger(__name__)


class Portfolio(metaclass=ABCMeta):
    """
    Base class for all portfolios.

    Any portfolio MUST inherit from this class an implement the following methods:
        * update_signal(self, event)
        * update_fill(self, event)

    Child portfolio classes must also call super().__init__() in order to set
    the class up correctly.
    """

    # stores all of the ticks portfolio position.
    POSITION_COLLECTION = 'portfolio'
    # stores the latest tick portfolio position.
    TICK_COLLECTION = 'portfolio_tick'

    def __init__(self,
                 data_handler: DataHandler,
                 events: queue.Queue,
                 start_date: datetime,
                 blotter: 'Blotter',
                 initial_capital: float = 100_000.00,
                 raise_on_warnings=False):
        self.logger = logging.getLogger(__name__)
        self.bars: DataHandler = data_handler
        self.events: queue.Queue = events
        self.blotter: 'Blotter' = blotter
        self.start_date: datetime = utils.parse_date(start_date)
        self._initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.ticker_list: List[str] = self.bars.tickers
        self.owned_assets: Dict[str, OwnedAsset] = {}
        # holdings = mv
        self.all_holdings_mv = self._construct_all_holdings()
        # positions = qty
        self.all_positions_qty = self._construct_all_positions()
        self.total_commission: float = 0.0
        self.lib: PortfolioStore = ARCTIC_STORE['pytech.portfolio']
        self.positions_df: pd.DataFrame = pd.DataFrame()
        self.raise_on_warnings: bool = raise_on_warnings
        self._signal_handlers = None

    @property
    def signal_handlers(self):
        return self._signal_handlers

    @signal_handlers.setter
    def signal_handlers(self, val: Iterable):
        if val is None:
            self._signal_handlers = []
        elif utils.is_iterable(val):
            self._signal_handlers = val
        else:
            raise TypeError('signal_handlers must be an iterable. '
                            f'{type(val)} was given.')

    @property
    def initial_capital(self):
        """Read only because it is just used to store this data for later."""
        return self._initial_capital

    @property
    def total_value(self) -> float:
        """
        A read only property to make getting the current total market value
        easier.
        **This includes cash.**
        """
        return self.total_asset_mv + self.cash

    @property
    def total_asset_mv(self) -> float:
        """
        A read only property to make getting the total market value of the
        owned assets easier.
        :return: The total market value of the owned assets in the portfolio.
        """
        mv = 0.0
        for asset in self.owned_assets.values():
            mv += asset.total_position_value
        return mv

    @abstractmethod
    def update_signal(self, event):
        """
        Acts on a :class:`SignalEvent` to generate new orders based on the
        portfolio logic.
        """
        raise NotImplementedError('Must implement update_signal()')

    @abstractmethod
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings based on a
        :class:`FillEvent`.
        """
        raise NotImplementedError('Must implement update_fill()')

    def _construct_all_positions(self):
        """
        Constructs the position list using the start date to determine when
        the index will begin.

        This should only be called once.
        """
        d = self._get_temp_dict()
        d['datetime'] = self.start_date
        return [d]

    def _construct_all_holdings(self):
        d = {k: v for k, v in [(ticker, 0.0) for ticker in self.ticker_list]}
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def _construct_current_holdings(self):
        """
        Construct a dict which holds the instantaneous market value of the
        portfolio across all symbols.

        This should only be called once.
        """
        d = {k: v for k, v in [(ticker, 0.0) for ticker in self.ticker_list]}
        return d

    def create_equity_curve_df(self):
        """Create a df from all_holdings_mv list of dicts."""
        curve = pd.DataFrame(self.all_holdings_mv)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve

    def _get_temp_dict(self):
        return {k: v for k, v in [(ticker, 0) for ticker in self.ticker_list]}

    def check_liquidity(self, avg_price_per_share, qty):
        """
        Check if the portfolio has enough liquidity to actually make the trade.
        This method should be called before
        executing any trade.

        :param float avg_price_per_share: The price per share in the trade
        **AFTER** commission has been applied.
        :param int qty: The amount of shares to be traded.
        :return: True if there is enough cash to make the trade or if qty is
        negative indicating a sale.
        """
        if qty < 0:
            return True

        cost = avg_price_per_share * qty
        cur_cash = self.cash
        post_trade_cash = cur_cash - cost

        return post_trade_cash > 0

    def get_owned_asset_mv(self, ticker: str) -> OwnedAsset:
        """
        Return the current market value for an :class:`OwnedAsset`

        :param ticker: The ticker of the owned asset.
        :return: The current market value for the ticker.
        :raises: KeyError
        """
        try:
            return self.owned_assets[ticker].total_position_value
        except KeyError:
            raise KeyError(f'Ticker: {ticker} is not currently owned.')

    def current_weights(self, include_cash: bool = False) -> Dict[str, float]:
        """
        Create a dictionary of the `portfolio`'s current weights at single
        point in time.

        :param include_cash: if cash should be included in determining the
        weights.
        :return: a dict with the key=ticker and value=weight.
        """
        weights = {}
        if include_cash:
            total_mv = self.total_value
            weights['cash'] = self.cash / total_mv
        else:
            total_mv = self.total_asset_mv

        for ticker, asset in self.owned_assets.items():
            weights[ticker] = asset.total_position_value / total_mv

        return weights

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for all the current market
        data bar. This reflects the PREVIOUS bar.
        Makes use of MarketEvent from the events queue.

        :param MarketEvent event:
        :raises InvalidEventTypeError: When the event type passed in is not
        a :class:`MarketEvent`
        """
        if event.event_type is not EventType.MARKET:
            raise InvalidEventTypeError(expected=EventType.MARKET,
                                        event_type=event.event_type)

        # get an element from the set
        latest_dt = self.bars.get_latest_bar_dt(next(iter(self.ticker_list)))
        # update the blotter's current date
        self.blotter.current_dt = latest_dt
        self.blotter.check_order_triggers()

        # update positions
        # dp = self._get_temp_dict()
        # dp['datetime'] = latest_dt
        dh = self._get_temp_dict()

        for ticker in self.ticker_list:
            try:
                dh[ticker] = self.owned_assets[ticker].shares_owned
            except KeyError:
                dh[ticker] = 0

        # append current positions
        # self.all_positions_qty.append(dp)

        # update holdings
        # dh = self._get_temp_dict()
        index = []
        dh['cash'] = self.cash
        dh['commission'] = self.total_commission
        dh['total'] = self.cash

        for ticker in self.ticker_list:
            try:
                owned_asset = self.owned_assets[ticker]
            except KeyError:
                market_value = 0
                self.logger.debug(f'{ticker} is not currently owned, '
                                  f'market value will be set to 0.')
            else:
                shares_owned = owned_asset.shares_owned
                adj_close = self.bars.latest_bar_value(ticker,
                                                       utils.ADJ_CLOSE_COL)
                market_value = shares_owned * adj_close
                owned_asset.update_total_position_value(adj_close, latest_dt)

            # approximate to real value.
            dh[ticker] = market_value
            dh['total'] += market_value
            index.append((latest_dt, ticker))

        multi_index = pd.MultiIndex.from_tuples(index, names=['datetime',
                                                              'ticker'])
        df = pd.DataFrame(dh, index=multi_index)

        self.positions_df = pd.concat([self.positions_df, df])
        self.logger.info('Writing current portfolio state to DB.')
        self.lib.write_snapshot(self.POSITION_COLLECTION,
                                self.positions_df,
                                latest_dt)
        self.lib.write_snapshot(self.TICK_COLLECTION, df, latest_dt)
        self.all_holdings_mv.append(dh)


class BasicPortfolio(Portfolio):
    """Here for testing and stuff."""

    def __init__(self,
                 data_handler: DataHandler,
                 events: queue.Queue,
                 start_date: datetime,
                 blotter: 'Blotter',
                 initial_capital: float = 100000.00,
                 raise_on_warnings=False):
        super().__init__(data_handler,
                         events,
                         start_date,
                         blotter,
                         initial_capital,
                         raise_on_warnings)

    def _update_from_trade(self, trade: Trade):
        self.cash += trade.trade_cost()
        self.total_commission += trade.commission

        if trade.ticker in self.owned_assets:
            self._update_existing_owned_asset_from_trade(trade)
        else:
            self._create_new_owned_asset_from_trade(trade)

    def _update_existing_owned_asset_from_trade(self, trade):
        """
        Update an existing owned asset or delete it if the trade results
        in all shares being sold.
        """
        owned_asset = self.owned_assets[trade.ticker]
        updated_asset = owned_asset.make_trade(trade.qty,
                                               trade.avg_price_per_share)

        if updated_asset is None:
            del self.owned_assets[trade.ticker]
        else:
            self.owned_assets[trade.ticker] = updated_asset

    def _create_new_owned_asset_from_trade(self, trade):
        """
        Create a new owned asset and add it to `self.owned_assets`, based on
        the execution of a trade.
        """
        if trade.action is TradeAction.SELL:
            asset_position = Position.SHORT
        else:
            asset_position = Position.LONG

        self.owned_assets[trade.ticker] = OwnedAsset.from_trade(trade,
                                                                asset_position)

    def update_fill(self, event):
        if event.type is EventType.FILL:
            order = self.blotter[event.order_id]
            if self.check_liquidity(event.price, event.available_volume):
                trade = self.blotter.make_trade(order,
                                                event.price,
                                                event.dt,
                                                event.available_volume)
                self._update_from_trade(trade)
            else:
                self.logger.warning(
                    'Insufficient funds available to execute trade for '
                    f'ticker: {order.ticker}')
                if self.raise_on_warnings:
                    raise InsufficientFundsError(ticker=order.ticker)

    def update_signal(self, event: SignalEvent) -> None:
        if event.event_type is EventType.SIGNAL:
            triggered_orders = self.blotter.check_order_triggers()
            self._process_signal(event, triggered_orders)
            # self.balancer(self, event)
            # self.events.put(self.generate_naive_order(event))
        else:
            raise InvalidEventTypeError(
                expected=type(EventType.SIGNAL),
                event_type=type(event.event_type))

    def _process_signal(self, signal: SignalEvent,
                        triggered_orders: List['AnyOrder']) -> None:
        """
        Call different methods depending on the type of signal received.

        :param signal:
        :return:
        """
        for handler in self.signal_handlers:
            handler.handle_signal(self, signal, triggered_orders)


class Portfolio_(object):
    """
    Holds stocks and keeps tracks of the owner's cash as well and their :class:`OwnedAssets` and allows them to perform
    analysis on just the :class:`Asset` that they currently own.


    Ignore this part:

    This class inherits from :class:``AssetUniverse`` so that it has access to its analysis functions.  It is important to
    note that :class:``Portfolio`` is its own table in the database as it represents the :class:``Asset`` that the user
    currently owns.  An :class:``Asset`` can be in both the *asset_universe* table as well as the *portfolio* table but
    a :class:``Asset`` does have to be in the database to be traded
    """

    LOGGER_NAME = 'portfolio'

    def __init__(self, starting_cash=1000000):
        """
        :param datetime start_date: a date, the start date to start the simulation as of.
            (default: end_date - 365 days)
        :param datetime end_date: the end date to end the simulation as of.
            (default: ``datetime.now()``)
        :param str benchmark_ticker: the ticker of the market index or benchmark to compare the portfolio against.
            (default: *^GSPC*)
        :param float starting_cash: the amount of dollars to allocate to the portfolio initially
            (default: 10000000)
        :param str trading_cal: The name of the trading calendar to use.
            (default: NYSE)
        :param data_frequency: The frequency of how often data should be updated.
        """

        self.owned_assets = {}
        self.cash = float(starting_cash)
        self.logger = logging.getLogger(self.LOGGER_NAME)

    def __getitem__(self, key):
        """Allow quick dictionary like access to the owned_assets dict"""

        if isinstance(key, OwnedAsset):
            return self.owned_assets[key.ticker]
        else:
            return self.owned_assets[key]

    def __setitem__(self, key, value):
        """Allow quick adding of :class:``OwnedAsset``s to the dict."""

        if isinstance(key, OwnedAsset):
            self.owned_assets[key.ticker] = value
        else:
            self.owned_assets[key] = value

        self.owned_assets[key] = value

    def __iter__(self):
        """
        Iterate over all the :class:`~.owned_asset.OwnedAsset`s
        in the portfolio.
        """
        yield self.owned_assets.items()

    def check_liquidity(self, avg_price_per_share, qty):
        """
        Check if the portfolio has enough liquidity to actually make the trade. This method should be called before
        executing any trade.

        :param float avg_price_per_share: The price per share in the trade
            **AFTER** commission has been applied.
        :param int qty: The amount of shares to be traded.
        :return: True if there is enough cash to make the trade or if qty is negative indicating a sale.
        """
        if qty < 0:
            return True

        cost = avg_price_per_share * qty
        cur_cash = self.cash
        post_trade_cash = cur_cash - cost

        return post_trade_cash > 0

    def update_from_trade(self, trade):
        """
        Update the ``portfolio``'s state based on the execution of a trade.  This includes updating the cash position
        as well as the ``owned_asset`` dictionary.

        :param Trade trade: The trade that was executed.
        :return:
        """
        self.cash += trade.trade_cost()

        if trade.ticker in self.owned_assets:
            self._update_existing_owned_asset_from_trade(trade)
        else:
            self._create_new_owned_asset_from_trade(trade)

    def _update_existing_owned_asset_from_trade(self, trade):
        """
        Update an existing owned asset or delete it if the trade results
        in all shares being sold.
        """
        owned_asset = self.owned_assets[trade.ticker]
        updated_asset = owned_asset.make_trade(trade.qty,
                                               trade.avg_price_per_share)

        if updated_asset is None:
            del self.owned_assets[trade.ticker]
        else:
            self.owned_assets[trade.ticker] = updated_asset

    def _create_new_owned_asset_from_trade(self, trade):
        """Create a new owned asset based on the execution of a trade."""

        if trade.action is TradeAction.SELL:
            asset_position = Position.SHORT
        else:
            asset_position = Position.LONG

        self.owned_assets[trade.ticker] = OwnedAsset.from_trade(trade,
                                                                asset_position)

    def get_total_value(self, include_cash=True):
        """
        Calculate the total value of the ``Portfolio`` owned_assets

        :param bool include_cash: Should cash be included in the calculation, or just get the total value of the
            owned_assets.
        :return: The total value of the portfolio at a given moment in time.
        :rtype: float
        """

        total_value = 0.0

        for asset in self.owned_assets.values():
            asset.update_total_position_value()
            total_value += asset.total_position_value

        if include_cash:
            total_value += self.cash

        return total_value

    def return_on_owned_assets(self):
        """Get the total return of the portfolio's current owned assets"""

        roi = 0.0
        for asset in self.owned_assets.values():
            roi += asset.return_on_investment()
        return roi

    def sma(self):
        for ticker, stock in self.owned_assets.items():
            yield stock.simple_moving_average()

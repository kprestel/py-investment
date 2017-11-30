import logging
import queue
import uuid
from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import (
    Dict,
    Iterable,
    List,
    TYPE_CHECKING,
)

import pandas as pd

import pytech.utils as utils
from pytech.backtest.event import (
    FillEvent,
    SignalEvent,
)
from pytech.data.connection import write
from pytech.data.handler import DataHandler
from pytech.fin.asset.owned_asset import OwnedAsset
from pytech.utils import DateRange

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
from pytech.exceptions import (
    InsufficientFundsError,
    InvalidEventTypeError,
    PyInvestmentTypeError,
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
                 date_range: DateRange,
                 blotter: 'Blotter',
                 initial_capital: float = 100_000.00,
                 raise_on_warnings=False) -> None:
        self._id = uuid.uuid4()
        self.date_range = date_range or DateRange()
        self.logger = logging.getLogger(__name__)
        self.bars: DataHandler = data_handler
        self.events: queue.Queue = events
        self.blotter: 'Blotter' = blotter
        self._initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.ticker_list: List[str] = self.bars.tickers
        self.owned_assets: Dict[str, OwnedAsset] = {}
        self.total_commission: float = 0.0
        self.positions_df: pd.DataFrame = pd.DataFrame()
        self.raise_on_warnings: bool = raise_on_warnings
        self.signal_handlers = None
        self.writer = write()
        self.writer.insert_portfolio(self)

    def __iter__(self):
        """
        Iterate over the owned_assets.
        """
        for asset in self.owned_assets.values():
            yield asset

    @property
    def id(self):
        """Returns a ``hex`` representation of the portfolios ``id``"""
        return self._id.hex

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
            raise PyInvestmentTypeError('signal_handlers must be an iterable. '
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

    @property
    def equity(self) -> float:
        """
        The total amount of equity the portfolio owns at a point in time.
        :return:
        """
        return sum(a.total_position_value for a in self)

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

    def check_liquidity(self, avg_price_per_share: float, qty: int) -> bool:
        """
        Check if the portfolio has enough liquidity to actually make the trade.
        This method should be called before
        executing any trade.

        :param float avg_price_per_share: The price per share in the trade
            **AFTER** commission has been applied.
        :param int qty: The amount of shares to be traded.
        :return: ``True`` if there is enough cash to make the trade or if qty
            is negative indicating a sale.
        """
        if qty < 0:
            return True

        cost = avg_price_per_share * qty
        post_trade_cash = self.cash - cost

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

    def get_asset_weight(self, ticker: str,
                         include_cash: bool = False) -> float:
        """
        Return the current weight an asset accounts for in the ``portfolio``.

        :param ticker: the ticker to get the weight for.
        :param include_cash: if cash should be included in determining the
            weights.
        :return: the weight as a float. If the asset is not owned then return
            0.
        """
        try:
            return self.current_weights(include_cash=include_cash)[ticker]
        except KeyError:
            return 0.0

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
        self.logger.debug(f'latest_dt={latest_dt}')
        # update the blotter's current date
        self.blotter.current_dt = latest_dt
        self.blotter.check_order_triggers()

        for owned_asset in self:
            adj_close = self.bars.latest_bar_value(owned_asset.ticker,
                                                   utils.ADJ_CLOSE_COL)
            owned_asset.update_total_position_value(adj_close, latest_dt)

        self.logger.debug('Writing current portfolio state to DB.')
        self.writer.portfolio_snapshot(self, latest_dt)
        if self.owned_assets:
            self.writer.owned_asset_snapshot(self.owned_assets.values(),
                                             self.id,
                                             latest_dt)


class BasicPortfolio(Portfolio):
    """Here for testing and stuff."""

    def __init__(self,
                 data_handler: DataHandler,
                 events: queue.Queue,
                 date_range: DateRange,
                 blotter: 'Blotter',
                 initial_capital: float = 100000.00,
                 raise_on_warnings=False):
        super().__init__(data_handler,
                         events,
                         date_range,
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

    def update_fill(self, event: FillEvent):
        if not event.event_type == EventType.FILL:
            raise InvalidEventTypeError(expected=type(EventType.SIGNAL),
                                        event_type=type(event.event_type))

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

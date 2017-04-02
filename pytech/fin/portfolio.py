import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime
from math import floor

import pandas as pd

import pytech.utils.pandas_utils as pd_utils
from pytech.backtest.event import FillEvent, TradeEvent, SignalEvent
from pytech.fin.owned_asset import OwnedAsset
from pytech.trading.trade import Trade
from pytech.utils.enums import EventType, OrderType, Position, SignalType, \
    TradeAction

logger = logging.getLogger(__name__)


class AbstractPortfolio(metaclass=ABCMeta):
    """
    Base class for all portfolios.

    Any portfolio MUST inherit from this class an implement the following methods:
        * update_signal(self, event)
        * update_fill(self, event)

    Child portfolio classes must also call super().__init__() in order to set the class up correctly.
    """

    def __init__(self, data_handler, events, start_date, blotter,
                 initial_capital=100000.00):
        self.logger = logging.getLogger(__name__)
        self.bars = data_handler
        self.events = events
        self.blotter = blotter
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.ticker_list = self.bars.ticker_list
        self.current_positions = self._get_temp_dict()
        # holdings = mv
        self.all_holdings = self.construct_all_holdings()
        # positions = qty
        self.all_positions = self.construct_all_positions()
        self.current_holdings = self.construct_current_holdings()
        self.total_commission = 0.0
        self.total_value = 0.0

    @abstractmethod
    def update_signal(self, event):
        """Acts on a :class:`SignalEvent` to generate new orders based on the portfolio logic."""

        raise NotImplementedError('Must implement update_signal()')

    @abstractmethod
    def update_fill(self, event):
        """Updates the portfolio current positions and holdings based on a :class:`FillEvent`"""

        raise NotImplementedError('Must implement update_fill()')

    def construct_all_positions(self):
        """Constructs the position list using the start date to determine when the index will begin"""

        d = self._get_temp_dict()
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self):
        d = {k: v for k, v in [(ticker, 0.0) for ticker in self.ticker_list]}
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        """
        Construct a dict which holds the instantaneous value of the 
        portfolio across all symbols.
        """

        d = {k: v for k, v in [(ticker, 0.0) for ticker in self.ticker_list]}
        return d

    def create_equity_curve_df(self):
        """Create a df from all_holdings list of dicts."""

        curve = pd.DataFrame(self.all_holdings)
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


class NaivePortfolio(AbstractPortfolio):
    """Here for testing and stuff."""

    def __init__(self, data_handler, events, start_date, blotter,
                 initial_capital=100000):
        super().__init__(data_handler, events, start_date, blotter,
                         initial_capital)

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for all the current market 
        data bar. This reflects the PREVIOUS bar.
        Makes use of MarketEvent from the events queue.

        :param MarketEvent event:
        :return:
        """

        self.blotter.check_order_triggers()

        # get an element from the set
        latest_dt = self.bars.get_latest_bar_dt(next(iter(self.ticker_list)))

        # update positions
        dp = self._get_temp_dict()
        dp['datetime'] = latest_dt

        for ticker in self.ticker_list:
            dp[ticker] = self.current_positions[ticker]

        # append current positions

        self.all_positions.append(dp)

        # update holdings
        dh = self._get_temp_dict()
        dh['datetime'] = latest_dt
        dh['cash'] = self.cash
        dh['commission'] = self.total_commission
        dh['total'] = self.cash

        for ticker in self.ticker_list:
            # approximate to real value.
            market_value = (self.current_positions[ticker] *
                            self.bars.get_latest_bar_value(ticker,
                                                           pd_utils.ADJ_CLOSE_COL))
            dh[ticker] = market_value
            dh['total'] += market_value

        self.all_holdings.append(dh)

    def update_positions_from_fill(self, trade):
        """
        Takes a :class:`Trade` and updates the position matrix to reflect new the position.
        :param Trade trade:
        :return:
        """

        self.current_positions[trade.ticker] += trade.qty

    def update_holdings_from_fill(self, trade):
        """
        Update the holdings matrix to reflect the holdings value.

        :param Trade trade:
        :return:
        """

        self.current_holdings[trade.ticker] += trade.trade_value()
        self.total_commission += trade.commission
        self.cash += trade.trade_value()
        self.total_value += trade.trade_value()

    def update_fill(self, event):

        if event.type is EventType.FILL:
            order = self.blotter[event.order_id]
            if self.check_liquidity(event.price, event.available_volume):
                trade = self.blotter.make_trade(order, event.price, event.dt,
                                                event.available_volume)
                self.update_positions_from_fill(trade)
                self.update_holdings_from_fill(trade)
            else:
                self.logger.warning(
                    'Insufficient funds available to execute trade for '
                    'ticker: {} '
                    .format(order.ticker))

    def generate_naive_order(self, signal):
        """
        Transacts an TradeEvent as a constant qty

        :param SignalEvent signal:
        :return:
        """

        mkt_qty = floor(100 * signal.strength)
        cur_qty = self.current_positions[signal.ticker]

        if signal.signal_type is SignalType.LONG and cur_qty == 0:
            return TradeEvent(signal.ticker, OrderType.MARKET, mkt_qty,
                              TradeAction.BUY)
        elif signal.signal_type is SignalType.SHORT and cur_qty == 0:
            return TradeEvent(signal.ticker, OrderType.MARKET, mkt_qty,
                              TradeAction.SELL)
        elif signal.signal_type is SignalType.EXIT and cur_qty > 0:
            return TradeEvent(signal.ticker, OrderType.MARKET, abs(cur_qty),
                              TradeAction.SELL)
        elif signal.signal_type is SignalType.EXIT and cur_qty < 0:
            return TradeEvent(signal.ticker, OrderType.MARKET, abs(cur_qty),
                              TradeAction.BUY)
        else:
            raise ValueError('Invalid signal. Cannot Create an order.')

    def update_signal(self, event):

        if event.type is EventType.SIGNAL:
            self.process_signal(event)
            self.blotter.check_order_triggers()
            # self.events.put(self.generate_naive_order(event))
        else:
            raise TypeError(
                'Invalid EventType. Must be EventType.SIGNAL. {} was provided'.format(
                    type(event)))

    def process_signal(self, signal):
        """
        Call different methods depending on the type of signal received.

        :param SignalEvent signal:
        :return:
        """

        if signal.signal_type is SignalType.EXIT:
            self.handle_exit_signal(signal)
        elif signal.signal_type is SignalType.CANCEL:
            self.handle_cancel_singal(signal)
        elif signal.signal_type is SignalType.HOLD:
            self.handle_hold_signal(signal)
        elif signal.signal_type not in [SignalType.LONG, SignalType.SHORT]:
            raise TypeError(
                'Invalid EventType. Must be EventType.SIGNAL. {} was provided'
                .format(type(signal.signal_type)))
        else:
            self.handle_long_short_signal(signal)

    def handle_long_short_signal(self, signal):
        """
        Create an order based on the **long** or **short** signal.

        :param SignalEvent signal:
        :return:
        """

    def handle_exit_signal(self, signal):
        """
        Create an order that will close out the position in the signal.

        :param SignalEvent signal:
        :return:
        """

    def handle_cancel_singal(self, signal):
        """
        Cancel all open orders for the asset in the signal.

        :param SignalEvent signal:
        :return:
        """

    def handle_hold_signal(self, signal):
        """
        Place all open orders for the asset in the signal on hold.

        :param SignalEvent signal:
        :return:
        """


class Portfolio(object):
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
        """Iterate over all the :class:`~.owned_asset.OwnedAsset`s in the portfolio."""

        yield self.owned_assets.items()

    def check_liquidity(self, avg_price_per_share, qty):
        """
        Check if the portfolio has enough liquidity to actually make the trade. This method should be called before
        executing any trade.

        :param float avg_price_per_share: The price per share in the trade **AFTER** commission has been applied.
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

        self.cash += trade.trade_value()

        if trade.ticker in self.owned_assets:
            self._update_existing_owned_asset_from_trade(trade)
        else:
            self._create_new_owned_asset_from_trade(trade)

    def _update_existing_owned_asset_from_trade(self, trade):
        """Update an existing owned asset or delete it if the trade results in all shares being sold."""

        updated_asset = self.owned_assets[trade.ticker].make_trade(trade.qty,
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

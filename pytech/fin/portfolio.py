import logging
import pandas as pd
from datetime import datetime
from abc import ABCMeta, abstractmethod
from math import floor

from pytech.fin.owned_asset import OwnedAsset
from pytech.backtest.event import SignalEvent, FillEvent, OrderEvent
from pytech.trading.trade import Trade
from pytech.utils.enums import TradeAction, Position, EventType, OrderType, SignalType

logger = logging.getLogger(__name__)


class AbstractPortfolio(metaclass=ABCMeta):

    @abstractmethod
    def update_signal(self, event):
        """Acts on a :class:`SignalEvent` to generate new orders based on the portfolio logic."""

        raise NotImplementedError('Must implement update_signal()')

    @abstractmethod
    def update_fill(self, event):
        """Updates the portfolio current positions and holdings based on a :class:`FillEvent`"""

        raise NotImplementedError('Must implement update_fill()')


class NaivePortfolio(AbstractPortfolio):
    """Here for testing and stuff."""

    def __init__(self, data_handler, events, start_date, initial_capital=100000):

        self.data_handler = data_handler
        self.events = events
        self.ticker_list = self.data_handler.ticker_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.all_positions = self.construct_all_positions()
        self.current_positions = {k: v for k, v in [(ticker, 0) for ticker in self.ticker_list]}

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

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
        """Construct a dict which holds the instantaneous value of the portfolio across all symbols."""

        d = {k: v for k, v in [(ticker, 0.0) for ticker in self.ticker_list]}
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        """
        Adds a new record tot eh positions matrix for all the current market data bar. This reflects the PREVIOUS bar.
        Makes use of MarketEvent from the events queue.

        :param MarketEvent event:
        :return:
        """

        bars = {}

        for ticker in self.ticker_list:
            bars[ticker] = self.data_handler.get_latest_bars(ticker)

        # update positions
        dp = self._get_temp_dict()
        dp['datetime'] = bars[self.ticker_list[0][0][1]]

        for ticker in self.ticker_list:
            dp[ticker] = self.current_positions[ticker]

        # append current positions

        self.all_positions.append(dp)

        # update holdings
        dh = self._get_temp_dict()
        dh['datetime'] = bars[self.ticker_list[0][0][1]]
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for ticker in self.ticker_list:
            # approximate to real value.
            market_value = self.current_holdings[ticker] * bars[ticker][0][5]
            dh[ticker] = market_value
            dh['total'] += market_value

        self.all_holdings.append(dh)

    def update_positions_from_fill(self, fill):
        """
        Takes a :class:`FillEvent` and updates the position matrix to reflect new the position.
        :param FillEvent fill:
        :return:
        """

        if fill.action is TradeAction.BUY:
            fill_dir = 1
        else:
            fill_dir = -1

        self.current_positions[fill.ticker] += fill_dir * fill.qty

    def update_holdings_from_fill(self, fill):
        """
        Update the holdings matrix to reflect the holdings value.

        :param FillEvent fill:
        :return:
        """

        if fill.action is TradeAction.BUY:
            fill_dir = 1
        else:
            fill_dir = -1

        fill_cost = self.data_handler.get_latest_bars(fill.ticker)[0][5]
        cost = fill_dir * fill_cost * fill.qty
        self.current_holdings[fill.ticker] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= cost + fill.commission
        self.current_holdings['total'] -= cost + fill.commission

    def update_fill(self, event):

        if event.type is EventType.FILL:
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)

    def generate_naive_order(self, signal):
        """
        Transacts an OrderEvent as a constant qty

        :param SignalEvent signal:
        :return:
        """

        mkt_qty = floor(100 * signal.strength)
        cur_qty = self.current_positions[signal.ticker]

        if signal.signal_type is SignalType.LONG and cur_qty == 0:
            return OrderEvent(signal.ticker, OrderType.MARKET, mkt_qty, TradeAction.BUY)
        elif signal.signal_type is SignalType.SHORT and cur_qty == 0:
            return OrderEvent(signal.ticker, OrderType.MARKET, mkt_qty, TradeAction.SELL)
        elif signal.signal_type is SignalType.EXIT and cur_qty > 0:
            return OrderEvent(signal.ticker, OrderType.MARKET, abs(cur_qty), TradeAction.SELL)
        elif signal.signal_type is SignalType.EXIT and cur_qty < 0:
            return OrderEvent(signal.ticker, OrderType.MARKET, abs(cur_qty), TradeAction.BUY)
        else:
            raise ValueError('Invalid signal. Cannot Create an order.')

    def update_signal(self, event):

        if event.type is EventType.SIGNAL:
            self.events.put(self.generate_naive_order(event))
        else:
            raise TypeError('Invalid EventType. Must be EventType.SIGNAL. {} was provided'.format(type(event)))

    def create_equity_curve_df(self):
        """Create a df from all_holdings list of dicts.l"""

        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve

    def _get_temp_dict(self):
        return {k: v for k, v in [(ticker, 0) for ticker in self.ticker_list]}


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

        updated_asset = self.owned_assets[trade.ticker].make_trade(trade.qty, trade.avg_price_per_share)

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

        self.owned_assets[trade.ticker] = OwnedAsset.from_trade(trade, asset_position)

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

from collections import namedtuple
import numbers
from datetime import datetime

import pandas as pd
from pandas_datareader import data as web

from pytech.fin.asset import Asset
from pytech.utils import dt_utils as dt_utils
from pytech.utils.enums import Position
from pytech.trading.trade import Trade
from pytech.utils.exceptions import NotAnAssetError


class OwnedAsset(object):
    """
    Contains data that only matters for a :class:`Asset` that is 
    in a user's :class:`~pytech.portfolio.Portfolio`.
    """

    def __init__(self, ticker, shares_owned, position,
                 average_share_price=None, purchase_date=None):
        self._ticker = ticker
        self.position = Position.check_if_valid(position)

        if purchase_date is None:
            self.purchase_date = pd.Timestamp(datetime.now())
        else:
            self.purchase_date = dt_utils.parse_date(purchase_date)

        if average_share_price:
            self.average_share_price_paid = average_share_price
            self.latest_price = average_share_price
            self.latest_price_time = self.purchase_date.time()

        self._shares_owned = shares_owned
        self._set_position_cost_and_value(qty=shares_owned,
                                          price=self.average_share_price_paid)

    @property
    def shares_owned(self):
        return self._shares_owned

    @shares_owned.setter
    def shares_owned(self, shares_owned):
        if isinstance(shares_owned, numbers.Integral):
            self._shares_owned = shares_owned
        else:
            raise TypeError('shares_owned MUST be an integer.')

    @property
    def ticker(self):
        if issubclass(self._ticker.__class__, Asset):
            return self._ticker.ticker
        else:
            return self._ticker

    @ticker.setter
    def ticker(self, ticker):
        if isinstance(ticker.__class__, Asset):
            self._ticker = ticker.ticker
        else:
            self._ticker = ticker

    @classmethod
    def from_trade(cls, trade, asset_position):
        """
        Create an owned_asset from a :class:``pytech.trading.trade.Trade``.  
        This the preferred method to create new ``OwnedStock`` objects, 
        and :func:``make_trade`` is the preferred way to update an instance of 
        :class:`OwnedStock`.

        :param Trade trade: The trade that will create the new instance of 
        :class:`OwnedStock`.
        :param Position asset_position: The position that the asset is, 
        either **LONG** or **SHORT**.
        :return: The newly created instance of ``OwnedStock`` to be added 
        to the owner's :class:``pytech.fin.portfolio``
        :rtype: OwnedAsset
        """
        owned_asset_dict = {
            'ticker': trade.ticker,
            'position': asset_position,
            'shares_owned': trade.qty,
            'average_share_price': trade.avg_price_per_share,
            'purchase_date': trade.trade_date
        }

        return cls(**owned_asset_dict)

    def make_trade(self, qty, price_per_share):
        """
        Update the position of the :class:`Stock`

        :param qty: int, positive if buying more shares and negative if selling shares
        :param price_per_share: float, the average price per share in the trade
        :return: self
        """
        self.shares_owned += qty
        self._set_position_cost_and_value(qty=qty, price=price_per_share)

        try:
            self.average_share_price_paid = (
                self.total_position_value / self.shares_owned)
        except ZeroDivisionError:
            return None
        else:
            return self

    def _set_position_cost_and_value(self, qty, price):
        """
        Calculate a position's cost and value

        :param qty: number of shares
        :type qty: int
        :param price: price per share
        :type price: long
        """
        if self.position is Position.SHORT:
            # short positions should have a negative number of shares owned
            # but a positive total cost
            self.total_position_cost = (price * qty) * -1
            # but a negative total value
            self.total_position_value = price * qty
        else:
            self.total_position_cost = price * qty
            self.total_position_value = (price * qty) * -1

    def update_total_position_value(self, latest_price, price_date):
        """
        Set the ``latest_price`` and ``latest_price_time`` and 
        update the total position's value

        :param latest_price:
        :param price_date:
        :return:
        """
        self.latest_price = latest_price
        self.latest_price_time = dt_utils.parse_date(price_date)
        if self.position is Position.SHORT:
            self.total_position_value = (
                                            self.latest_price * self.shares_owned) * -1
        else:
            self.total_position_value = self.latest_price * self.shares_owned

    def return_on_investment(self):
        """
        Get the current return on investment for a given 
        :class:`OwnedAsset`
        """
        self.update_total_position_value()
        return (self.total_position_value + self.total_position_cost) / (
            self.total_position_cost * -1)

    def market_correlation(self, use_portfolio_benchmark=True,
                           market_ticker='^GSPC'):
        """
        Compute the correlation between a :class: Stock's return and the market return.
        :param use_portfolio_benchmark:
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker:
            Any valid ticker symbol to use as the market.
        :return:

        Best used to gauge the accuracy of the beta.
        """

        pct_change = self._get_pct_change(
                use_portfolio_benchmark=use_portfolio_benchmark,
                market_ticker=market_ticker)
        return pct_change.stock_pct_change.corr(pct_change.market_pct_change)

    def calculate_beta(self, use_portfolio_benchmark=True,
                       market_ticker='^GSPC'):
        """
        Compute the beta for the :class: Stock

        :param use_portfolio_benchmark: boolean
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker:
            Any valid ticker symbol to use as the market.
        :return: float
            The beta for the given Stock
        """
        pct_change = self._get_pct_change(
                use_portfolio_benchmark=use_portfolio_benchmark,
                market_ticker=market_ticker)
        covar = pct_change.stock_pct_change.cov(pct_change.market_pct_change)
        variance = pct_change.market_pct_change.var()
        return covar / variance

    def _get_pct_change(self, use_portfolio_benchmark=True,
                        market_ticker='^GSPC'):
        """
        Get the percentage change over the :class: Stock's start and end dates for both the asset as well as the market

        :param use_portfolio_benchmark: boolean
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker: str
            Any valid ticker symbol to use as the market.
        :return: TimeSeries
        """
        pct_change = namedtuple('Pct_Change',
                                'market_pct_change stock_pct_change')
        if use_portfolio_benchmark:
            market_df = self.portfolio.benchmark
        else:
            market_df = web.DataReader(market_ticker, 'yahoo',
                                       start=self.start_date,
                                       end=self.end_date)
        market_pct_change = pd.Series(
                market_df['adj_close'].pct_change(periods=1))
        stock_pct_change = pd.Series(
                self.ohlcv['adj_close'].pct_change(periods=1))
        return pct_change(market_pct_change=market_pct_change,
                          stock_pct_change=stock_pct_change)

    def _get_portfolio_benchmark(self):
        """
        Helper method to get the :class: Portfolio's benchmark ticker symbol
        :return: TimeSeries
        """

        return self.portfolio.benchmark

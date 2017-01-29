# from pytech import Session
import logging
from datetime import datetime, timedelta

import pandas_datareader.data as web

import pytech.db.db_utils as db
import pytech.utils as utils
from pytech.blotter import Blotter

logger = logging.getLogger(__name__)


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

    def __init__(self, start_date=None, end_date=None, benchmark_ticker='^GSPC', starting_cash=1000000,
                 trading_cal='NYSE', data_frequency='daily'):
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

        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=365)
        else:
            self.start_date = utils.parse_date(start_date)

        if end_date is None:
            # default to today
            self.end_date = datetime.now()
        else:
            self.end_date = utils.parse_date(end_date)

        self.trading_cal = trading_cal
        self.owned_assets = {}
        self.orders = {}
        self.benchmark_ticker = benchmark_ticker
        self.benchmark = web.DataReader(benchmark_ticker, 'yahoo', start=self.start_date, end=self.end_date)
        self.cash = float(starting_cash)
        self.data_frequency = data_frequency
        self.logger = logging.getLogger(self.LOGGER_NAME)
        self.blotter = Blotter(portfolio=self)

    def get_owned_asset(self, ticker):
        """
        Return the :py:class:`pytech.asset.OwnedAsset` instance if it exists.

        This method is also used to check if an asset is owned.

        :param str ticker: The ticker of the asset to get.
        :return: The ``OwnedAsset`` instance or None if it does not exist.
        """

        return self.owned_assets.get(ticker)


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

        with db.transactional_session() as session:
            # update the owned stocks in the db
            session.add(self)

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

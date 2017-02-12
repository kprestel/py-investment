from datetime import datetime
from datetime import timedelta

import pandas_market_calendars as mcal
from sqlalchemy import create_engine

from pytech.db.connector import DBConnector
from pytech.db.finders import AssetFinder

import pytech.utils.dt_utils as dt_utils


class Environment(object):
    """Set up the local trading environment settings"""

    def __init__(self, benchmark_sym='^GPSC', trading_tz='US/Eastern', trade_cal='NYSE', db_path=None):

        self.benchmark_sym = benchmark_sym
        self.trading_tz = trading_tz
        self.trade_cal = mcal.get_calendar(trade_cal)
        self.db_conn = DBConnector(db_path)
        self.db_conn.init_db()
        self.asset_finder = AssetFinder(self.db_conn.engine)


class SimParams(object):
    """Parameters that will be held constant throughout the entire simulation."""

    def __init__(self, trading_cal, start_date, end_date, data_frequency='daily'):

        self._trading_cal = trading_cal


        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=365)
        else:
            self.start_date = dt_utils.parse_date(start_date)

        if end_date is None:
            # default to today
            self.end_date = datetime.now()
        else:
            self.end_date = dt_utils.parse_date(end_date)

        self.data_frequency = data_frequency

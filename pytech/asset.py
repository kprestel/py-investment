import logging
import os
import re
from collections import namedtuple
from datetime import datetime, date, timedelta
from multiprocessing import Process

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from dateutil import parser
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from sqlalchemy import Column, Numeric, String, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy import orm
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import AbstractConcreteBase
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy_utils import generic_relationship
from twisted.internet import reactor

import pytech.db_utils as db
import pytech.utils as utils
from crawler.spiders.edgar import EdgarSpider
from pytech import DATA_DIR
from pytech.base import Base
from pytech.exceptions import InvalidPositionError, AssetNotInUniverseError, PyInvestmentError, NotAnAssetError
from pytech.enums import AssetPosition

logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioAsset(object):
    """
    Mixin object to create a one to many relationship with Portfolio

    Any class that inherits from this class may be added to a Portfolio's asset_list which will place a foreign key
    in that asset class and create a relationship between it and the Portfolio that owns it
    """

    @declared_attr
    def portfolio_id(cls):
        return Column('portfolio_id', ForeignKey('portfolio.id'))


class HasStock(object):
    """
    Mixin object to create a relation to a asset object

    Any class that inherits from this class will be given a foreign key column that corresponds to the asset object
    passed to the child class's constructor
    """

    @declared_attr
    def stock_id(cls):
        return Column('stock_id', ForeignKey('stock.id'))


class Asset(Base, AbstractConcreteBase):
    """
    This is the base class that all Asset classes should inherit from.

    Inheriting from it will provide a table name and the proper mapper args required for the db.  It will also allow it
    to have a relationship with the :class:``OwnedAsset``.

    The child class is responsible for giving each instance a ticker to identify it.

    If the child class needs any more fields it is responsible for creating them at the class level as well as
    populating them via the child's constructor, in addition to calling the ``Asset`` constructor.

    Any child class instance of this base class is considered to be a part of the **Asset Universe** or the assets that
    are eligible to be traded.  If a child instance of an Asset does not yet exist in the universe and the
    :class:``~pytech.portfolio.Portfolio`` tries to trade it an exception will occur.
    """

    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True)

    def __init__(self, ticker):
        self.ticker = ticker
        self.ohlcv = None
        self.file_path = os.path.join(DATA_DIR, (ticker + '.csv'))

    @declared_attr
    def __tablename__(cls):
        name = cls.__name__
        return (
            name[0].lower() +
            re.sub(r'([A-Z])',
                   lambda m: '_' + m.group(0).lower(), name[1:])
        )

    @declared_attr
    def __mapper_args__(cls):
        name = cls.__name__
        return {
            'polymorphic_identity': name.lower(),
            'concrete': True
        }

    @classmethod
    def get_asset_from_universe(cls, ticker):
        """
        Query the asset universe for the requested ticker and return the object

        :param ticker: the ticker of the ``Asset`` being traded
        :type ticker: str
        :return: The :class:``Asset`` object with the ticker passed in
        :rtype: Asset
        :raises AssetNotInUniverseError: if an :class:``Asset`` with the requested ticker cannot be found
        """

        with db.transactional_session() as session:
            asset = session.query(cls).filter(cls.ticker == ticker).first()
            if asset is not None:
                return asset
            else:
                raise AssetNotInUniverseError(ticker=ticker)

    def get_price_quote(self, d=None, column='Adj Close'):
        """
        Get the price of an Asset.

        :param date or datetime d: The datetime of when to retrieve the price quote from.
            (default: ``date.today()``)
        :param str column: The header of the column to use to get the price quote from.
            (default: ``Adj Close``
        :return: namedtuple with price and the the datetime
        :rtype: namedtuple
        """
        quote = namedtuple('Quote', 'price time')
        if d is None:
            df = web.get_quote_yahoo(self.ticker)
            d = date.today()
            time = utils.parse_date(df['time'][0]).time()
            dt = datetime.combine(d, time=time)
            return quote(price=df['last'], time=dt)
        else:
            price = self.ohlcv.ix[d][column][0]
            return quote(price=price, time=d)

    def get_volume(self, dt):
        """
        Get the current volume traded for the ``Asset``

        :param datetime dt: The datetime to get the volume for.
        :return: The volume for the ``Asset``
        :rtype: int
        """

        return self.ohlcv.ix[dt]['Volume'][0]


class Stock(Asset):
    """
    Main class that is used to model stocks and may contain technical and fundamental data about the asset.

    A ``Stock`` object must exist in the database in order for it to be traded.  Each ``Stock`` object is considered to
    be in the *universe* of owned_assets the portfolio owner is willing to own/trade
    """

    start_date = Column(DateTime)
    end_date = Column(DateTime)
    latest_price = Column(Numeric)
    latest_price_time = Column(DateTime)
    get_ohlcv = Column(Boolean)
    load_fundamentals = Column(Boolean)
    beta = Column(Numeric)
    start_price = Column(Numeric)
    end_price = Column(Numeric)
    fundamentals = relationship('Fundamental',
                                collection_class=attribute_mapped_collection('access_key'),
                                lazy='joined')

    def __init__(self, ticker, start_date=None, end_date=None, get_fundamentals=False, get_ohlcv=True):

        super().__init__(ticker)

        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=365)
        else:
            self.start_date = utils.parse_date(start_date)

        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = utils.parse_date(end_date)

        if self.start_date >= self.end_date:
            raise ValueError(
                    'start_date must be older than end_date. start_date: {} end_date: {}'.format(str(start_date),
                                                                                                 str(end_date)))
        self.get_ohlcv = get_ohlcv
        if get_ohlcv:
            self.get_ohlcv_series()
            self.beta = self.calculate_beta()
            self.start_price = self.ohlcv[['Adj Close']].head(1).iloc[0]['Adj Close']
            self.end_price = self.ohlcv[['Adj Close']].tail(1).iloc[0]['Adj Close']
        else:
            self.beta = None
            self.start_price = None
            self.end_price = None

        self.fundamentals = {}
        self.load_fundamentals = get_fundamentals
        if get_fundamentals:
            self.get_fundamentals()
        quote = self.get_price_quote()
        self.latest_price = quote.price
        self.latest_price_time = quote.time

    @orm.reconstructor
    def init_on_load(self):
        """If the user wanted the ohlc_series then recreate it when this object is loaded again"""
        self.get_ohlcv_series()
        # if self.get_ohlcv:
        #     self.get_ohlcv_series()
        # else:
        #     self.ohlcv = None
        quote = self.get_price_quote()
        self.latest_price = quote.price
        self.latest_price_time = quote.time

    def get_ohlcv_series(self, data_source='yahoo', start_date=None, end_date=None):
        """
        Load the ohlcv timeseries.

        :param str data_source: set where to get the data from. see pandas DataReader docs for more valid options.
            (default: yahoo)
        :param datetime start_date: When to load the timeseries as of.
        :param datetime end_date: When to end the timeseries.

        This method will get called on the initial creation of the :class:``Stock`` object with whatever start and end
        dates that the object is created with.

        To change the date range that is contained in the timeseries then call this method explicitly with the desired
        time period as arguments.
        """

        # TODO: concatenate the new series to any existing series

        def get_ohlcv_from_web(start_date, end_date, data_source):
            try:
                return web.DataReader(self.ticker, data_source=data_source, start=start_date, end=end_date)
            except:
                logger.exception('Could not create series for ticker: {}. Unknown error occurred.'.format(self.ticker))
                return None

        if start_date is not None:
            start_date = utils.parse_date(start_date)
        else:
            start_date = self.start_date

        if end_date is not None:
            end_date = utils.parse_date(end_date)
        else:
            end_date = self.end_date

        try:
            temp_df = pd.read_csv(self.file_path, parse_dates=['Date'])
            temp_df.set_index('Date')
        except IOError:
            # if the file_path does not exist then default to the web
            ohlcv = get_ohlcv_from_web(start_date=start_date, end_date=end_date, data_source=data_source)
            ohlcv.to_csv(self.file_path)
            self.ohlcv = ohlcv
        else:
            # TODO: get any missing data
            max_date = temp_df.index.max()
            min_date = temp_df.index.min()
            # if max_date < end_date:
            #     pass
            self.ohlcv = temp_df



    def get_fundamentals(self):
        """
        :return:

        pass the required attributes to the EdgarSpider and it will create the corresponding fundamentals objects
        for this asset instance as well as write the fundamental object to the DB
        """

        # session = Session()
        with db.query_session() as session:
            # check to see if the fundamentals already exist
            # NOTE: there may be an edge case where the start and end dates are different and so not all of the desired
            # fundamentals will be returned but for now this should work
            # it will require some sort of date guessing based on the periods and the end dates or something so yeah
            # that is a problem for another day
            result = session.query(Fundamental).filter(Fundamental.ticker == self.ticker).all()
            if result:
                for row in result:
                    self.fundamentals[row.access_key] = row
            else:
                self._init_spiders([self.ticker], end_date=self.end_date, start_date=self.start_date)
                result = session.query(Fundamental).filter(Fundamental.stock_id == self.id).all()
                for row in result:
                    self.fundamentals[row.access_key] = row

    # TECHNICAL INDICATORS/ANALYSIS

    def simple_moving_average(self, period=50, column='Adj Close'):
        table_name = 'sma_test'
        # stmt = text('SELECT * FROM sma_test WHERE asset_id = :asset_id')
        # stmt.bindparams(asset_id=self.id)
        sql = 'SELECT * FROM sma_test WHERE asset_id = :asset_id'
        conn = db.raw_connection()
        print(Base.metadata)
        try:
            # TODO: parse dates
            df = pd.read_sql(sql, con=conn, params={
                'asset_id': self.id
                })
        except OperationalError:
            logger.exception('error in query')
            sma_ts = pd.Series(
                    self.ohlcv[column].rolling(center=False, window=period, min_periods=period - 1).mean()).dropna()
            db.df_to_sql(sma_ts, 'sma_test', asset_id=self.id)
            print('creating')
            print(sma_ts)
            return sma_ts
            # return sma_ts
        else:
            print('found')
            print(df)
            return df

    def simple_moving_median(self, period=50, column='Adj Close'):
        """
        :param ohlc: dict
        :param period: int, the number of days to use
        :param column: string, the name of the column to use to compute the median
        :return: Timeseries containing the simple moving median

        compute the simple moving median over a given period and return it in timeseries
        """
        return pd.Series(self.ohlcv[column].rolling(center=False, window=period, min_periods=period - 1).median(),
                         name='{} day SMM Ticker: {}'.format(period, self.ticker))

    def exponential_weighted_moving_average(self, period=50, column='Adj Close'):
        """
        :param ohlc: dict
        :param period: int, the number of days to use
        :param column: string, the name of the column to use to compute the mean
        :return: Timeseries containing the simple moving median

        compute the exponential weighted moving average (ewma) over a given period and return it in timeseries
        """
        return pd.Series(self.ohlcv[column].ewm(ignore_na=False, min_periods=period - 1, span=period).mean(),
                         name='{} day EWMA Ticker: {}'.format(period, self.ticker))

    def double_ewma(self, period=50, column='Adj Close'):
        """

        :param self: Stock
        :param period: int, days
        :param column: string
        :return: generator

        double exponential moving average
        """
        ewma = self._ewma_computation(period=period, column=column)
        ewma_mean = ewma.ewm(ignore_na=False, min_periods=period - 1, span=period).mean()
        dema = 2 * ewma - ewma_mean
        yield pd.Series(dema, name='{} day DEMA Ticker: {}'.format(period, self.ticker))

    def triple_ewma(self, period=50, column='Adj Close'):
        """
        :param self: Stock
        :param period: int, days
        :param column: string
        :return: generator

        triple exponential moving average
        """
        ewma = self._ewma_computation(period=period, column=column)
        triple_ema = 3 * ewma
        ema_ema_ema = ewma.ewm(ignore_na=False, span=period).mean().ewm(ignore_na=False, span=period).mean()
        tema = triple_ema - 3 * ewma.ewm(ignore_na=False, min_periods=period - 1, span=period).mean() + ema_ema_ema
        return pd.Series(tema, name='{} day TEMA Ticker: {}'.format(period, self.ticker))

    def triangle_moving_average(self, period=50, column='Adj Close'):
        """
        :param self: dict
        :param period: int, days
        :param column: string
        :return: time series

        triangle moving average

        SMA of the SMA
        """
        sma = self._sma_computation(period=period, column=column) \
            .rolling(center=False, window=period, min_periods=period - 1).mean()
        return pd.Series(sma, name='{} day TRIMA Ticker: {}'.format(period, self.ticker))

    def triple_ema_oscillator(self, period=15, column='Adj Close'):
        """
        :param universe_dict: dict
        :param period: int, days
        :param column: string
        :return: generator

        triple exponential moving average oscillator (trix)

        calculates the triple smoothed EMA of n periods and finds the pct change between 1 period of EMA3

        oscillates around 0. positive numbers indicate a bullish indicator
        """
        emwa_one = self._ewma_computation(period=period, column=column)
        emwa_two = emwa_one.ewm(ignore_na=False, min_periods=period - 1, span=period).mean()
        emwa_three = emwa_two.ewm(ignore_na=False, min_periods=period - 1, span=period).mean()
        trix = emwa_three.pct_change(periods=1)
        return pd.Series(trix, name='{} days TRIX Ticker: {}'.format(period, self.ticker))

    def efficiency_ratio(self, period=10, column='Adj Close'):
        """
        :param universe_dict: dict
        :param period: int, days
        :param column: string
        :return: generator

        Kaufman Efficiency Indicator. oscillates between +100 and -100

        positive is bullish
        """
        change = self.ohlcv[column].diff(periods=period).abs()
        vol = self.ohlcv[column].diff().abs().rolling(window=period).sum()
        return pd.Series(change / vol, name='{} days Efficiency Indicator Ticker: {}'.format(period, self.ticker))

    def _efficiency_ratio_computation(self, period=10, column='Adj Close'):
        """
        :param ohlc: Timeseries
        :param period: int, days
        :param column: string
        :return: Timeseries

        Kaufman Efficiency Indicator. oscillates between +100 and -100

        positive is bullish
        """

        change = self.ohlcv[column].diff(periods=period).abs()
        vol = self.ohlcv[column].diff().abs().rolling(window=period).sum()
        return pd.Series(change / vol)

    def kama(self, efficiency_ratio_periods=10, ema_fast=2, ema_slow=30, period=20, column='Adj Close'):
        er = self._efficiency_ratio_computation(period=efficiency_ratio_periods, column=column)
        fast_alpha = 2 / (ema_fast + 1)
        slow_alpha = 2 / (ema_slow + 1)
        smoothing_constant = pd.Series((er * (fast_alpha - slow_alpha) + slow_alpha) ** 2, name='smoothing_constant')
        sma = pd.Series(self.ohlcv[column].rolling(period).mean(), name='SMA')
        kama = []
        for smooth, ma, price in zip(iter(smoothing_constant.items()), iter(sma.shift(-1).items()),
                                     iter(self.ohlcv[column].items())):
            try:
                kama.append(kama[-1] + smooth[1] * (price[1] - kama[-1]))
            except:
                if pd.notnull(ma[1]):
                    kama.append(ma[1] + smooth[1] * (price[1] - ma[1]))
                else:
                    kama.append(None)
        sma['KAMA'] = pd.Series(kama, index=sma.index, name='{} days KAMA Ticker {}'.format(period, self.ticker))
        yield sma['KAMA']

    def zero_lag_ema(self, period=30, column='Adj Close'):
        """
        :param universe_dict: dict
        :param period: int, days
        :param column: string
        :return: generator

        zero lag exponential moving average

        """
        lag = (period - 1) / 2
        return pd.Series((self.ohlcv[column] + (self.ohlcv[column].diff(lag))),
                         name='{} days Zero Lag EMA Ticker: {}'.format(period, self.ticker))

    def weighted_moving_average(self, period=30, column='Adj Close'):
        """
        :param universe_dict: dict
        :param period: int, days
        :param column: string
        :return: generator

        aims to smooth the price curve for better trend identification
        places a higher importance on recent data compared to the EMA
        """
        wma = self._weighted_moving_average_computation(period=period, column=column)
        # ts['WMA'] = pd.Series(wma, index=ts.index)
        return pd.Series(pd.Series(wma, index=self.ohlcv.index),
                         name='{} days WMA Ticker: {}'.format(period, self.ticker))
        # yield pd.Series(ts['WMA'], name='{} days WMA Ticker: {}'.format(period, ticker))

    def hull_moving_average(self, period=30, column='Adj Close'):
        """

        :param universe_dict: dict
        :param period: int, days
        :param column: string
        :return: generator

        smoother than the SMA, it aims to minimize lag and track price trends more accurately

        best used in mid to long term analysis
        """
        import math
        wma_one_period = int(period / 2) * 2
        wma_one = pd.Series(self._weighted_moving_average_computation(period=wma_one_period, column=column),
                            index=self.ohlcv.index)
        wma_one *= 2
        wma_two = pd.Series(self._weighted_moving_average_computation(period=period, column=column),
                            index=self.ohlcv.index)
        wma_delta = wma_one - wma_two
        sqrt_period = int(math.sqrt(period))
        wma = self._weighted_moving_average_computation(period=sqrt_period, column=column)
        wma_delta['_WMA'] = pd.Series(wma, index=self.ohlcv.index)
        yield pd.Series(wma_delta['_WMA'], name='{} day HMA Ticker: {}'.format(period, self.ticker))

    def volume_weighted_moving_average(universe_dict, period=30, column='Adj Close'):
        pass

    def smoothed_moving_average(self, period=30, column='Adj Close'):
        """
        :param universe_dict: dict
        :param period: int, days
        :param column: string
        :return: generator

        equal weights given to historic and more current prices
        """
        return pd.Series(self.ohlcv[column].ewm(alpha=1 / float(period)).mean(),
                         name='{} days SMMA Ticker: {}'.format(period, self.ticker))

    def macd_signal(self, period_fast=12, period_slow=26, signal=9, column='Adj Close'):
        """
        Moving average convergence divergence

        :param universe_dict: dict
        :param period_fast: int, traditionally 12
        :param period_slow: int, traditionally 26
        :param signal: int, traditionally 9
        :param column: string
        :return:

        signals:
            when the MACD falls below the signal line this is a bearish signal, and vice versa
            when security price diverages from MACD it signals the end of a trend
            if MACD rises dramatically quickly, the shorter moving averages pulls away from the slow moving average
            it is a signal that the security is overbought and should come back to normal levels soon

        as with any signals this can be misleading and should be combined with something to avoid being faked out

        NOTE: be careful changing the default periods, the method wont break but this is the 'traditional' way of doing this

        """

        ema_fast = pd.Series(
                self.ohlcv[column].ewm(ignore_na=False, min_periods=period_fast - 1, span=period_fast).mean(),
                name='EMA_fast')
        ema_slow = pd.Series(
                self.ohlcv[column].ewm(ignore_na=False, min_periods=period_slow - 1, span=period_slow).mean(),
                name='EMA_slow')
        macd_series = pd.Series(ema_fast - ema_slow, name='MACD')
        macd_signal_series = pd.Series(macd_series.ewm(ignore_na=False, span=signal).mean(), name='MACD_Signal')
        return pd.concat([macd_signal_series, macd_series], axis=1)

    def market_momentum(self, period=10, column='Adj Close'):
        """
        Continually take price differences for a fixed interval

        Positive or negative number plotted on a zero line

        :param universe_dict: dict
        :param period: int
        :param column: string
        :return: generator
        """

        return pd.Series(self.ohlcv[column].diff(period), name='{} day MOM Ticker: {}'.format(period, self.ticker))

    def rate_of_change(self, period=1, column='Adj Close'):
        """
        :param universe_dict: dict
        :param period: int
        :param column: string
        :return: generator

        simply calculates the rate of change between two periods
        """
        return pd.Series((self.ohlcv[column].diff(period) / self.ohlcv[column][-period]) * 100,
                         name='{} day Rate of Change Ticker: {}'.format(period, self.ticker))

    def relative_strength_indicator(self, period=14, column='Adj Close'):
        """
        :param universe_dict: dict
        :param period: int
        :param column: string
        :return: generator

        RSI oscillates between 0 and 100 and traditionally +70 is considered overbought and under 30 is oversold
        """
        return pd.Series(self._rsi_computation(period=period, column=column),
                         name='{} day RSI Ticker: {}'.format(period, self.ticker))

    def inverse_fisher_transform(self, rsi_period=5, wma_period=9, column='Adj Close'):
        """
        Modified Inverse Fisher Transform applied on RSI

        :param universe_dict: dict
        :param rsi_period: int, period that is used for the RSI calculation
        :param wma_period: int, period that is used for the WMA RSI calculation
        :param column: string
        :return: generator

        Buy when indicator crosses -0.5 or crosses +0.5
        RSI is smoothed with WMA before applying the transformation

        IFT_RSI signals buy when the indicator crosses -0.5 or crosses +0.5 if it has not previously crossed over -0.5
        it signals to sell short when indicators crosses under +0.5 or crosses under -0.5 if it has not previously crossed +.05
        """
        import numpy as np
        v1 = pd.Series(.1 * (self._rsi_computation(period=rsi_period, column=column) - 50),
                       name='v1')
        v2 = pd.Series(self._weighted_moving_average_computation(ts=v1, period=wma_period, column=column),
                       index=v1.index)
        return pd.Series((np.exp(2 * v2) - 1) / (np.exp(2 * v2) + 1),
                         name='{} day IFT_RSI Ticker: {}'.format(rsi_period, self.ticker))

    def true_range(self, period=14):
        """
        :param universe_dict: dict
        :param period: int
        :return: generator

        finds the true range a asset is trading within
        most recent period's high - most recent periods low
        absolute value of the most recent period's high minus the previous close
        absolute value of the most recent period's low minus the previous close

        this will give you a dollar amount that the asset's range that it has been trading in
        """
        # TODO: make this method use adjusted close
        range_one = pd.Series(self.ohlcv['High'].tail(period) - self.ohlcv['Low'].tail(period), name='high_low')
        range_two = pd.Series(self.ohlcv['High'].tail(period) - self.ohlcv['Close'].shift(-1).abs().tail(period),
                              name='high_prev_close')
        range_three = pd.Series(self.ohlcv['Close'].shift(-1).tail(period) - self.ohlcv['Low'].abs().tail(period),
                                name='prev_close_low')
        tr = pd.concat([range_one, range_two, range_three], axis=1)
        true_range_list = []
        for row in tr.itertuples():
            # TODO: fix this so it doesn't throw an exception for weekends
            try:
                true_range_list.append(max(row.high_low, row.high_prev_close, row.prev_close_low))
            except TypeError:
                continue
        tr['TA'] = true_range_list
        return pd.Series(tr['TA'], name='{} day TR Ticker: {}'.format(period, self.ticker))

    def average_true_range(self, period=14):
        """
        Moving average of a asset's true range
        :param period: int
        :return: generator
        """
        tr = self._true_range_computation(period=period * 2)
        return pd.Series(tr.rolling(center=False, window=period, min_periods=period - 1).mean(),
                         name='{} day ATR Ticker: {}'.format(period, self.ticker)).tail(period)

    def bollinger_bands(self, period=30, moving_average=None, column='Adj Close'):
        std_dev = self.ohlcv[column].std()
        if isinstance(moving_average, pd.Series):
            middle_band = pd.Series(self._sma_computation(period=period, column=column),
                                    name='middle_bband')
        else:
            middle_band = pd.Series(moving_average, name='middle_bband')

        upper_bband = pd.Series(middle_band + (2 * std_dev), name='upper_bband')
        lower_bband = pd.Series(middle_band - (2 * std_dev), name='lower_bband')

        percent_b = pd.Series((self.ohlcv[column] - lower_bband) / (upper_bband - lower_bband), name='%b')
        b_bandwidth = pd.Series((upper_bband - lower_bband) / middle_band, name='b_bandwidth')
        return pd.concat([upper_bband, middle_band, lower_bband, b_bandwidth, percent_b], axis=1)

    def _get_portfolio_benchmark(self):
        """
        Helper method to get the :class: Portfolio's benchmark ticker symbol
        :return: TimeSeries
        """
        from pytech.portfolio import Portfolio

        with db.query_session() as session:
            benchmark_ticker = \
                session.query(Portfolio.benchmark_ticker) \
                    .filter(Portfolio.id == self.portfolio_id) \
                    .first()
        if not benchmark_ticker:
            return web.DataReader('^GSPC', 'yahoo', start=self.start_date, end=self.end_date)
        else:
            return web.DataReader(benchmark_ticker, 'yahoo', start=self.start_date, end=self.end_date)

    def _get_pct_change(self, market_ticker='^GSPC'):
        """
        Get the percentage change over the :class: Stock's start and end dates for both the asset as well as the market

        :param use_portfolio_benchmark: boolean
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker: str
            Any valid ticker symbol to use as the market.
        :return: namedtuple
        """
        pct_change = namedtuple('Pct_Change', 'market_pct_change stock_pct_change')
        try:
            market_df = web.DataReader(market_ticker, 'yahoo', start=self.start_date, end=self.end_date)
        except:
            raise Exception('Unknown error occurred trying to get a OHLCV for ticker: {}'.format(market_ticker))
        market_pct_change = pd.Series(market_df['Adj Close'].pct_change(periods=1))
        stock_pct_change = pd.Series(self.ohlcv['Adj Close'].pct_change(periods=1))
        return pct_change(market_pct_change=market_pct_change, stock_pct_change=stock_pct_change)

    def calculate_beta(self, market_ticker='^GSPC'):
        """
        Compute the beta for the :class: Stock

        :param use_portfolio_benchmark: boolean
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker:
            Any valid ticker symbol to use as the market.
        :return: float
            The beta for the given Stock
        """
        pct_change = self._get_pct_change(market_ticker=market_ticker)
        covar = pct_change.stock_pct_change.cov(pct_change.market_pct_change)
        variance = pct_change.market_pct_change.var()
        return covar / variance

    def market_correlation(self, market_ticker='^GSPC'):
        """
        Compute the correlation between a :class: Stock's return and the market return.
        :param use_portfolio_benchmark:
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker:
            Any valid ticker symbol to use as the market.
        :return:

        Best used to gauge the accuracy of the beta.
        """
        pct_change = self._get_pct_change(market_ticker=market_ticker)
        return pct_change.stock_pct_change.corr(pct_change.market_pct_change)

    def adj_return(self, risk_free_rate_ticker='TB1YR'):
        risk_free_rate = web.DataReader(risk_free_rate_ticker, 'fred', start=self.start_date, end=self.end_date) \
            .tail(1).iloc[0][risk_free_rate_ticker]
        stock_return = ((self.end_price - self.start_price) / self.start_price) * 100
        return stock_return - risk_free_rate

    def roi(self):
        """
        Compute the return on investment for a :class: Stock
        :return: float
        """
        return ((self.end_price - self.start_price) / self.start_price) * 100

    def directional_movement_indicator(self, period=14):
        """
        :param universe_dict: dict
        :param period: int
        :return: Series generator

        DMI also known as Average Directional Movement Index (ADX)

        this is a lagging indicator that only indicates a trend's strength rather than trend direction
        so it is best coupled with another movement indicator to determine the strength of a trend

        a strategy created by Alexander Elder states a buy signal is triggered when the DMI peaks and starts to decline
        when the positive dmi is above the negative dmi. a sell signal is triggered when dmi stops falling and goes flat
        """
        temp_df = pd.DataFrame()
        temp_df['up_move'] = self.ohlcv['High'].diff()
        temp_df['down_move'] = self.ohlcv['Low'].diff()

        positive_dm = []
        negative_dm = []

        for row in temp_df.itertuples():
            if row.up_move > row.down_move and row.up_move > 0:
                positive_dm.append(row.up_move)
            else:
                positive_dm.append(0)
            if row.down_move > row.up_move and row.down_move > 0:
                negative_dm.append(row.down_move)
            else:
                negative_dm.append(0)
        temp_df['positive_dm'] = positive_dm
        temp_df['negative_dm'] = negative_dm
        atr = self._average_true_range_computation(ts=self.ohlcv, period=period * 6)
        diplus = pd.Series(100 * (temp_df['positive_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(),
                           name='positive_dmi')
        diminus = pd.Series(100 * (temp_df['negative_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(),
                            name='negative_dmi')
        return pd.concat([diplus, diminus])

    def sma_crossover_signals(self, slow=200, fast=50, column='Adj Close'):
        """
        :param slow: int, how many days for the short term moving average
        :param fast:  int, how many days for the long term moving average
        :param column: str
        :return:
        """
        slow_ts = self.simple_moving_average(period=slow, column=column)
        fast_ts = self.simple_moving_average(period=fast, column=column)
        crossover_ts = pd.Series(fast_ts - slow_ts, name='test', index=self.ohlcv.index)
        # if 50 SMA > 200 SMA set action to 1 which means Buy
        # TODO: figure out a better way to mark buy vs sell
        # also need to make sure this method works right...
        self.ohlcv['Action'] = np.where(crossover_ts > 0, 1, 0)

    def simple_median_crossover_signals(self, slow=200, fast=50, column='Adj Close'):
        slow_ts = self.simple_moving_median(period=slow, column=column)
        fast_ts = self.simple_moving_median(period=fast, column=column)
        crossover_ts = pd.Series(fast_ts - slow_ts, name='test', index=self.ohlcv.index)
        crossover_ts['Action'] = np.where(crossover_ts > 0, 1, 0)
        print(crossover_ts)

    # Util Methods

    def _directional_movement_indicator(self, period):
        """
        :param ts: Series
        :param period: int
        :return: Series

        DMI also known as average directional index
        """
        temp_df = pd.DataFrame()
        temp_df['up_move'] = self.ohlcv['High'].diff()
        temp_df['down_move'] = self.ohlcv['Low'].diff()

        positive_dm = []
        negative_dm = []

        for row in temp_df.itertuples():
            if row.up_move > row.down_move and row.up_move > 0:
                positive_dm.append(row.up_move)
            else:
                positive_dm.append(0)
            if row.down_move > row.up_move and row.down_move > 0:
                negative_dm.append(row.down_move)
            else:
                negative_dm.append(0)
        temp_df['positive_dm'] = positive_dm
        temp_df['negative_dm'] = negative_dm
        atr = self._average_true_range_computation(period=period * 6)
        diplus = pd.Series(100 * (temp_df['positive_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(),
                           name='positive_dmi')
        diminus = pd.Series(100 * (temp_df['negative_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(),
                            name='negative_dmi')
        return pd.concat([diplus, diminus])

    def _true_range_computation(self, period):
        """
        :param period: int
        :return: Timeseries

        this method is used internally to compute the average true range of a asset

        the purpose of having it as separate function is so that external functions can return generators
        """
        range_one = pd.Series(self.ohlcv['High'].tail(period) - self.ohlcv['Low'].tail(period), name='high_low')
        range_two = pd.Series(self.ohlcv['High'].tail(period) - self.ohlcv['Close'].shift(-1).abs().tail(period),
                              name='high_prev_close')
        range_three = pd.Series(self.ohlcv['Close'].shift(-1).tail(period) - self.ohlcv['Low'].abs().tail(period),
                                name='prev_close_low')
        tr = pd.concat([range_one, range_two, range_three], axis=1)
        true_range_list = []
        for row in tr.itertuples():
            # TODO: fix this so it doesn't throw an exception for weekends
            try:
                true_range_list.append(max(row.high_low, row.high_prev_close, row.prev_close_low))
            except TypeError:
                continue
        tr['TA'] = true_range_list
        return pd.Series(tr['TA'])

    def _sma_computation(self, period=50, column='Adj Close'):
        return pd.Series(self.ohlcv[column].rolling(center=False, window=period, min_periods=period - 1).mean())

    def _average_true_range_computation(self, period):
        tr = self._true_range_computation(period=period * 2)
        return pd.Series(tr.rolling(center=False, window=period, min_periods=period - 1).mean())

    def _rsi_computation(self, period, column):
        """
        :param period: int
        :param column: string
        :return: Series
        :rtype: TimeSeries

        relative strength indicator
        """
        ts = self.ohlcv
        gain = [0]
        loss = [0]
        for row, shifted_row in zip(iter(ts[column].items()), iter(ts[column].shift(-1).items())):
            if row[1] - shifted_row[1] > 0:
                gain.append(row[1] - shifted_row[1])
                loss.append(0)
            elif row[1] - shifted_row[1] < 0:
                gain.append(0)
                loss.append(abs(row[1] - shifted_row[1]))
            elif row[1] - shifted_row[1] == 0:
                gain.append(0)
                loss.append(0)
        # TODO: make this a copy so it doesnt change the original ts
        ts['gain'] = gain
        ts['loss'] = loss

        avg_gain = ts['gain'].rolling(window=period).mean()
        avg_loss = ts['loss'].rolling(window=period).mean()
        relative_strength = avg_gain / avg_loss
        return pd.Series(100 - (100 / (1 + relative_strength)))

    def _weighted_moving_average_computation(self, period, column, ts=None):
        wma = []
        if ts is None:
            ts = self.ohlcv
        for chunk in self._chunks(period=period, column=column, ts=ts):
            # TODO: figure out a better way to handle this. this is better than a catch all except though
            try:
                wma.append(self.chunked_weighted_moving_average(chunk=chunk, period=period))
            except AttributeError:
                wma.append(None)
        wma.reverse()
        return wma

    def _chunks(self, period, column, ts=None):
        """
        :param ts: Timeseries
        :param period: int, the amount of chunks needed
        :param column: string
        :return: generator

        creates n chunks based on the number of periods
        """
        if ts is None:
            ts = self.ohlcv
        # reverse the ts
        try:
            ts_rev = ts[column].iloc[::-1]
        except KeyError:
            ts_rev = ts.iloc[::-1]
        for i in enumerate(ts_rev):
            chunk = ts_rev.iloc[i[0]:i[0] + period]
            if len(chunk) != period:
                yield None
            else:
                yield chunk

    @staticmethod
    def _chunked_weighted_moving_average(chunk, period):
        """
        :param chunk: Timeseries, should be in chunks
        :param period: int, the number of chunks/days
        :return:
        """
        denominator = (period * (period + 1)) / 2
        ma = []
        for price, i in zip(chunk.iloc[::-1].tolist(), list(range(period + 1))[1:]):
            ma.append(price * (i / float(denominator)))
        return sum(ma)

    def _ewma_computation(self, period=50, column='Adj Close'):
        """
        this method is used for computations in other exponential moving averages

        :param ohlc: Timeseries
        :param period: int, number of days
        :param column: string
        :return: Timeseries
        """
        return pd.Series(self.ohlcv[column].ewm(ignore_na=False, min_periods=period - 1, span=period).mean())

    @staticmethod
    def _init_spiders(ticker_list, start_date, end_date):
        """
        Start a subprocess to run the spiders in.

        :param ticker_list: list of tickers to get fundamentals for
        :type ticker_list: list
        :param start_date: date to start scraping as of
        :type start_date: datetime
        :param end_date: date to stop scraping as of
        :type end_date: datetime

        The main reason behind  this method is to work around the fact that ``~twisted.reactor`` cannot be restarted
        so we start it in a subprocess that can be killed off.
        """
        p = Process(target=Stock._run_spiders, args=(ticker_list, start_date, end_date))
        p.start()
        p.join()

    @staticmethod
    def _run_spiders(ticker_list, start_date, end_date):
        configure_logging()
        runner = CrawlerRunner(settings=get_project_settings())

        spider_dict = {
            'symbols': ticker_list,
            'start_date': start_date,
            'end_date': end_date
        }
        runner.crawl(EdgarSpider, **spider_dict)
        d = runner.join()
        d.addBoth(lambda _: reactor.stop())
        reactor.run()

    """
    ALTERNATE CONSTRUCTORS
    """

    @classmethod
    def from_list(cls, ticker_list, start, end, get_ohlcv=False, get_fundamentals=False):
        """
        Create a dict of stocks for a given time period based on the list of ticker symbols passed in

        :param ticker_list: list of str
            must they must correspond to a valid ticker symbol. They will be used to create the :class: Stock objects
        :param start: date or str date formatted YYYYMMDD
            when to load the ohlcv and the :class: Fundamental as of
        :param end: date or str date formatted YYYYMMDD
            when to load the ohlcv and the :class: Fundamental as of
        :param get_ohlcv: boolean
            if true then a ohlc time series will be created based on the start and end dates
        :return: generator
            contains one :class:`Stock` per ticker in the ticker list
        """

        if get_fundamentals:
            cls._init_spiders(ticker_list=ticker_list, start_date=start, end_date=end)

        with db.transactional_session() as session:
            for ticker in ticker_list:
                session.add(cls(ticker=ticker, start_date=start, end_date=end, get_ohlcv=get_ohlcv,
                                get_fundamentals=get_fundamentals))

    @classmethod
    def from_dict(cls, stock_dict):
        d = {k: v for k, v in stock_dict.items() if k in cls.__dict__}
        return cls(**d)

    @classmethod
    def create_stocks_dict_from_list_and_write_to_db(cls, ticker_list, start, end, session=None, get_fundamentals=False,
                                                     get_ohlc=False, close_session=True):
        """
        :param ticker_list: list of str objects, they must correspond to a valid ticker symbol
        :param start: datetime, start date
        :param end: datetime, end date
        :param session: sqlalchemy session, if None then one will be created and closed
        :param get_fundamentals: boolean, if Fundamental objects should be created then set this to True
        :param get_ohlc: boolean, if true then an ohlc time series will bce created based on the start and end dates
        :param close_session: boolean, if a user passes a session in and they don't want it to be closed after then set
        this to false
        :return: dict, contains asset objects with their corresponding tickers as the key

        create a dict of stocks for a given time period based on the list of ticker symbols passed in
        """
        if session is None:
            from pytech import Session
            session = Session()
        if get_fundamentals:
            return cls.create_stocks_dict_with_fundamentals_from_list(ticker_list=ticker_list, start=start, end=end,
                                                                      session=session, get_ohlc=get_ohlc,
                                                                      close_session=close_session)
        stock_dict = {}
        for ticker in ticker_list:
            temp_stock = cls(ticker=ticker, start_date=start, end_date=end, get_ohlc=get_ohlc)
            stock_dict[ticker] = temp_stock
            session.add(temp_stock)
        session.commit()
        if close_session:
            session.close()
        return stock_dict

    @classmethod
    def create_stocks_dict_from_list(cls, ticker_list, start, end, get_fundamentals=False, get_ohlc=False,
                                     session=None):
        """
        :param session: sqlalchemy session, if None is provided then the stocks will not be written to the DB
        :param ticker_list: list of str objects that have corresponding tickers
        :param start: datetime
        :param end: datetime
        :param get_fundamentals: boolean
        :param get_ohlc: boolean
        :return: dict

        create a dictionary of asset objects with their tickers as keys and write them to the DB if a session is provided
        NOTE: if get_fundamentals is True then stocks have to be written to the DB
        """
        if get_fundamentals:
            return cls.create_stocks_dict_with_fundamentals_from_list(ticker_list=ticker_list, start=start, end=end,
                                                                      get_ohlc=get_ohlc, session=session)
        elif session is not None:
            return cls.create_stocks_dict_from_list_and_write_to_db(ticker_list=ticker_list, start=start, end=end,
                                                                    session=session, get_fundamentals=get_fundamentals,
                                                                    get_ohlc=get_ohlc)
        stock_dict = {}
        for ticker in ticker_list:
            stock_dict[ticker] = cls(ticker=ticker, start_date=start, end_date=end, get_ohlc=get_ohlc)
        return stock_dict


class OwnedAsset(Base):
    """
    Contains data that only matters for a :class:`Asset` that is in a user's :class:`~pytech.portfolio.Portfolio`
    """

    asset_id = Column(Integer)
    #: The type of asset it is. For example if for a Stock asset the **asset_type** would be 'stock'
    asset_type = Column(String)
    #: The relation to the :class:``Asset`` instance
    asset = generic_relationship(asset_id, asset_type)
    portfolio_id = Column(Integer, ForeignKey('portfolio.id'), primary_key=True)
    purchase_date = Column(DateTime)
    average_share_price_paid = Column(Numeric)
    shares_owned = Column(Integer)
    total_position_value = Column(Numeric)
    #: The total amount of capital invested in an asset.
    #: This will be negative for a long position and positive for a short position
    total_position_cost = Column(Numeric)
    #: Long or Short
    position = Column(String)

    def __init__(self, asset, portfolio, shares_owned, position, average_share_price=None, purchase_date=None):

        if issubclass(asset.__class__, Asset):
            self.asset = asset
        else:
            raise NotAnAssetError(asset=type(asset))

        self.portfolio = portfolio
        self.position = AssetPosition.check_if_valid(position)

        if purchase_date is None:
            self.purchase_date = datetime.now()
        else:
            self.purchase_date = utils.parse_date(purchase_date)

        if average_share_price:
            self.average_share_price_paid = average_share_price
            self.latest_price = average_share_price
            self.latest_price_time = self.purchase_date.time()
        else:
            quote = asset.get_price_quote()
            self.average_share_price_paid = quote.price
            self.latest_price = quote.price
            self.latest_price_time = quote.time

        self._shares_owned = shares_owned
        self._set_position_cost_and_value(qty=shares_owned, price=self.average_share_price_paid)

    @property
    def shares_owned(self):
        return self._shares_owned

    @shares_owned.setter
    def shares_owned(self, shares_owned):
        self._shares_owned = int(shares_owned)

    def make_trade(self, qty, price_per_share=None):
        """
        Update the position of the :class:`Stock`

        :param qty: int, positive if buying more shares and negative if selling shares
        :param price_per_share: float, the average price per share in the trade
        :return: self
        """

        self.shares_owned += qty

        if price_per_share:
            self._set_position_cost_and_value(qty=qty, price=price_per_share)
        else:
            quote = self.asset.get_price_quote()
            self.latest_price = quote.price
            self.latest_price_time = quote.time
            self._set_position_cost_and_value(qty=qty, price=quote.price)

        try:
            self.average_share_price_paid = self.total_position_value / float(self.shares_owned)
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
        if self.position is AssetPosition.SHORT:
            # short positions should have a negative number of shares owned but a positive total cost
            self.total_position_cost = (price * qty) * -1
            # but a negative total value
            self.total_position_value = price * qty
        else:
            self.total_position_cost = price * qty
            self.total_position_value = (price * qty) * -1

    def update_total_position_value(self):
        """Retrieve the latest market quote and update the ``OwnedStock`` attributes to reflect the change"""

        quote = self.asset.get_price_quote()
        self.latest_price = quote.price
        self.latest_price_time = quote.time
        if self.position is AssetPosition.SHORT:
            self.total_position_value = (self.latest_price * self.shares_owned) * -1
        else:
            self.total_position_value = self.latest_price * self.shares_owned

    def return_on_investment(self):
        """Get the current return on investment for a given :class:``OwnedAsset``"""

        self.update_total_position_value()
        return (self.total_position_value + self.total_position_cost) / (self.total_position_cost * -1)

    def market_correlation(self, use_portfolio_benchmark=True, market_ticker='^GSPC'):
        """
        Compute the correlation between a :class: Stock's return and the market return.
        :param use_portfolio_benchmark:
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker:
            Any valid ticker symbol to use as the market.
        :return:

        Best used to gauge the accuracy of the beta.
        """

        pct_change = self._get_pct_change(use_portfolio_benchmark=use_portfolio_benchmark, market_ticker=market_ticker)
        return pct_change.stock_pct_change.corr(pct_change.market_pct_change)

    def calculate_beta(self, use_portfolio_benchmark=True, market_ticker='^GSPC'):
        """
        Compute the beta for the :class: Stock

        :param use_portfolio_benchmark: boolean
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker:
            Any valid ticker symbol to use as the market.
        :return: float
            The beta for the given Stock
        """
        pct_change = self._get_pct_change(use_portfolio_benchmark=use_portfolio_benchmark, market_ticker=market_ticker)
        covar = pct_change.stock_pct_change.cov(pct_change.market_pct_change)
        variance = pct_change.market_pct_change.var()
        return covar / variance

    def _get_pct_change(self, use_portfolio_benchmark=True, market_ticker='^GSPC'):
        """
        Get the percentage change over the :class: Stock's start and end dates for both the asset as well as the market

        :param use_portfolio_benchmark: boolean
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker: str
            Any valid ticker symbol to use as the market.
        :return: TimeSeries
        """
        pct_change = namedtuple('Pct_Change', 'market_pct_change stock_pct_change')
        if use_portfolio_benchmark:
            market_df = self.portfolio.benchmark
        else:
            market_df = web.DataReader(market_ticker, 'yahoo', start=self.start_date, end=self.end_date)
        market_pct_change = pd.Series(market_df['Adj Close'].pct_change(periods=1))
        stock_pct_change = pd.Series(self.ohlcv['Adj Close'].pct_change(periods=1))
        return pct_change(market_pct_change=market_pct_change, stock_pct_change=stock_pct_change)

    def _get_portfolio_benchmark(self):
        """
        Helper method to get the :class: Portfolio's benchmark ticker symbol
        :return: TimeSeries
        """

        return self.portfolio.benchmark


class Fundamental(Base, HasStock):
    """
    The purpose of the this class to hold one period's worth of fundamental data for a given asset

    NOTE: If the period focus is Q1, Q2, Q3 then all cumulative measures will be QTD and if its FY they will be YTD

    Would it be a good idea to break this class out into period measures, like sales, expenses, etc and then instant
    measures like accounts pay, balance sheet stuff?
    """

    # key to the corresponding Stock's dictionary must be 'period_focus_year'
    id = Column(Integer, primary_key=True)
    access_key = Column(String, unique=True)
    amended = Column(Boolean)
    assets = Column(Numeric)
    current_assets = Column(Numeric)
    current_liabilities = Column(Numeric)
    cash = Column(Numeric)
    dividend = Column(Numeric)
    end_date = Column(DateTime)
    eps = Column(Numeric)
    eps_diluted = Column(Numeric)
    equity = Column(Numeric)
    net_income = Column(Numeric)
    operating_income = Column(Numeric)
    revenues = Column(Numeric)
    investment_revenues = Column(Numeric)
    fin_cash_flow = Column(Numeric)
    inv_cash_flow = Column(Numeric)
    ops_cash_flow = Column(Numeric)
    year = Column(String)
    period_focus = Column(String)
    property_plant_equipment = Column(Numeric)
    gross_profit = Column(Numeric)
    tax_expense = Column(Numeric)
    net_taxes_paid = Column(Numeric)
    acts_pay_current = Column(Numeric)
    acts_receive_current = Column(Numeric)
    acts_receive_noncurrent = Column(Numeric)
    acts_receive = Column(Numeric)
    accrued_liabilities_current = Column(Numeric)
    inventory_net = Column(Numeric)
    interest_expense = Column(Numeric)
    total_liabilities = Column(Numeric)
    total_liabilities_equity = Column(Numeric)
    shares_outstanding = Column(Numeric)
    shares_outstanding_diluted = Column(Numeric)
    depreciation_amortization = Column(Numeric)
    cogs = Column(Numeric)
    comprehensive_income_net_of_tax = Column(Numeric)
    research_and_dev_expense = Column(Numeric)
    common_stock_outstanding = Column(Numeric)
    warranty_accrual = Column(Numeric)
    warranty_accrual_payments = Column(Numeric)
    ebit = Column(Numeric)
    ebitda = Column(Numeric)
    ticker = Column(String)

    def __init__(self, amended, assets, current_assets, current_liabilities, cash, dividend, end_date, eps, eps_diluted,
                 equity, net_income, operating_income, revenues, investment_revenues, fin_cash_flow, inv_cash_flow,
                 ops_cash_flow, year, property_plant_equipment, gross_profit, tax_expense, net_taxes_paid,
                 acts_pay_current, acts_receive_current, acts_receive_noncurrent, accrued_liabilities_current,
                 period_focus, inventory_net, interest_expense, total_liabilities, total_liabilities_equity,
                 shares_outstanding, shares_outstanding_diluted, common_stock_outstanding, depreciation_amortization,
                 cogs, comprehensive_income_net_of_tax, research_and_dev_expense, warranty_accrual,
                 warranty_accrual_payments, ticker):
        """
        This :class: should never really be instantiated directly.  It is intended to be instantiated by the
        :class: EdgarSpider after it has finished scraping the XBRL page that it will be associated with
        because that is where the data will come from that populates the class.

        :param amended: str
            were the finical statements amended
        :param assets: float
            total owned_assets
        :param current_assets: float
            total current owned_assets
        :param current_liabilities: float
            total current liabilities
        :param cash: float
            total cash and cash equivalents
        :param dividend: float
            if a dividend was paid how much was it
        :param end_date: date
            end of the fiscal period
        :param eps: float
            earnings per share
        :param eps_diluted: float
            diluted earnings per share
        :param equity: float
            total equity
        :param net_income:
        :param operating_income:
        :param revenues:
        :param investment_revenues:
        :param fin_cash_flow:
        :param inv_cash_flow:
        :param ops_cash_flow:
        :param year:
        :param property_plant_equipment:
        :param gross_profit:
        :param tax_expense:
        :param net_taxes_paid:
        :param acts_pay_current:
        :param acts_receive_current:
        :param acts_receive_noncurrent:
        :param accrued_liabilities_current:
        :param period_focus:
        """
        self.amended = amended
        self.assets = assets
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.cash = cash
        self.dividend = dividend
        self.end_date = utils.parse_date(end_date)
        self.eps = eps
        self.eps_diluted = eps_diluted
        self.equity = equity
        self.net_income = net_income
        self.operating_income = operating_income
        self.revenues = revenues
        self.investment_revenues = investment_revenues
        self.fin_cash_flow = fin_cash_flow
        self.inv_cash_flow = inv_cash_flow
        self.ops_cash_flow = ops_cash_flow
        self.period_focus = period_focus
        self.year = year
        self.gross_profit = gross_profit
        self.property_plant_equipment = property_plant_equipment
        self.tax_expense = tax_expense
        self.net_taxes_paid = net_taxes_paid
        self.acts_pay_current = acts_pay_current
        self.acts_receive_current = acts_receive_current
        self.acts_receive_noncurrent = acts_receive_noncurrent
        if acts_receive_noncurrent is None or acts_receive_current is None:
            self.acts_receive = None
        else:
            self.acts_receive = acts_receive_noncurrent + acts_receive_current
        self.accrued_liabilities_current = accrued_liabilities_current
        self.access_key = str(year) + '_' + period_focus
        self.inventory_net = inventory_net
        self.interest_expense = interest_expense
        self.total_liabilities = total_liabilities
        self.total_liabilities_equity = total_liabilities_equity
        self.shares_outstanding = shares_outstanding
        self.shares_outstanding_diluted = shares_outstanding_diluted
        self.common_stock_outstanding = common_stock_outstanding
        self.depreciation_amortization = depreciation_amortization
        self.cogs = cogs
        self.comprehensive_income_net_of_tax = comprehensive_income_net_of_tax
        self.research_and_dev_expense = research_and_dev_expense
        self.warranty_accrual = warranty_accrual
        self.warranty_accrual_payments = warranty_accrual_payments
        self.ebit = self._ebit()
        self.ebitda = self._ebitda()
        self.ticker = ticker

    def return_on_assets(self):
        return self.net_income / self.assets

    def debt_ratio(self):
        return self.assets / self.total_liabilities

    # LIQUIDITY RATIOS

    def current_ratio(self):
        """
        Also known as working capital ratio
        :return:
        """
        return self.current_assets / self.current_liabilities

    def quick_ratio(self):
        """
        Liquidity ratio. Current ratio - inventory
        :return: float
        """
        return (self.current_assets - self.inventory_net) / self.current_liabilities

    def cash_ratio(self):
        """
        Most conservative of the liquidity ratios
        :return:
        """
        return self.cash / self.current_liabilities

    # METHODS FOR THE CONSTRUCTOR
    # There should not be any reason for you to call these methods directly because you can access the results directly
    # from the :class:`Fundamental` object itself

    def _ebitda(self):
        """
        Earnings before interest, tax, depreciation and amortization
        :return: float
        """
        try:
            return self.net_income + self.tax_expense + self.interest_expense + self.depreciation_amortization
        except TypeError:
            logger.exception('net_income: {}, tax_expense: {}, interest_expense: {}, depreciation_amortization: {}'
                             .format(self.net_income, self.tax_expense, self.interest_expense,
                                     self.depreciation_amortization))

    def _ebit(self):
        """
        Earnings before interest, tax
        :return: float
        """
        return self.net_income + self.tax_expense + self.interest_expense

    @classmethod
    def from_json_file(cls, stock, year, period_focus=None):
        """
        :param stock:
        :param year:
        :param period_focus:
        :return:

        this should probably just get deleted
        """
        import json
        if not isinstance(stock, Stock):
            raise TypeError('asset must be an instance of a asset object. {} was provided'.format(type(stock)))
        logger.info('Getting {} fundamental data'.format(stock.ticker))
        if period_focus is None:
            file_name = 'FY_{}.json'.format(stock.ticker)
        else:
            file_name = '{}_{}.json'.format(period_focus, stock.ticker)
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'financials', stock.ticker.upper())):
            os.mkdir(os.path.join(os.path.dirname(__file__), '..', 'financials', stock.ticker.upper()))
            base_file_path = os.path.join(os.path.dirname(__file__), '..', 'financials', stock.ticker.upper())
        else:
            base_file_path = os.path.join(os.path.dirname(__file__), '..', 'financials', stock.ticker.upper())
        file_path = os.path.join(base_file_path, year, file_name)
        with open(file_path) as f:
            data = json.load(f)
            logger.debug('{} loaded'.format(file_path))
        # was this report restated/amended
        amended = data.amend
        assets = data.assets
        current_assets = data.cur_assets
        current_liabilities = data.cur_liab
        cash = data.cash
        dividend = data.dividend
        # TODO: convert to date. need to test if all dates are the same format
        end_date = data.end_date
        eps = data.eps_basic
        eps_diluted = data.eps_diluted
        equity = data.equity
        net_income = data.net_income
        operating_income = data.op_income
        revenues = data.revenues
        investment_revenues = data.investment_revenues
        fin_cash_flow = data.cash_flow_fin
        inv_cash_flow = data.cash_flow_inv
        ops_cash_flow = data.cash_flow_op
        ticker = data.symbol
        return cls(amended=amended, assets=assets, current_assets=current_assets,
                   current_liabilities=current_liabilities,
                   cash=cash, dividend=dividend, end_date=end_date, eps=eps, eps_diluted=eps_diluted, equity=equity,
                   net_income=net_income, operating_income=operating_income, revenues=revenues,
                   investment_revenues=investment_revenues, fin_cash_flow=fin_cash_flow, inv_cash_flow=inv_cash_flow,
                   ops_cash_flow=ops_cash_flow, year=year, period_focus=period_focus, ticker=ticker)

    @classmethod
    def from_dict(cls, fundamental_dict):
        """
        This should be the only way the :class: `Fundamental` is instantiated.

        :param fundamental_dict: dict, created by the :class:`~crawler.spiders.EdgarSpider` which passes an item to the
            :class:`~crawler.item_pipeline.FundamentalItemPipeline` that creates the dict and then passes it to this method
        :return: :class:`Fundamental` object
        """
        df = {k: v for k, v in fundamental_dict.items() if k in cls.__dict__}
        return cls(**df)

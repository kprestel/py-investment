# from pytech import Session
from pytech.base import Base
from collections import namedtuple
import pandas as pd
import os
import numpy as np
import pandas_datareader.data as web
import datetime
import logging
import re
from dateutil import parser
from sqlalchemy import orm
from crawler.spiders.edgar import EdgarSpider
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging

from sqlalchemy.ext.declarative import as_declarative
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import Column, Numeric, String, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
import pytech.db_utils as db

logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# Base = Base()
# Base.metadata.create_all(db.engine)

class PortfolioAsset(object):
    """
    Mixin object to create a one to many relationship with Portfolio

    Any class that inherits from this class may be added to a Portfolio's asset_list which will place a foreign key
    in that asset class and create a relationship between it and the Portfolio that owns it
    """

    @declared_attr
    def portfolio_id(cls):
        return Column('portfolio_id', ForeignKey('portfolio.id'))

        # @declared_attr
        # def portfolio(cls):
        #     return relationship('Portfolio', )

        # @declared_attr
        # def portfolio(cls):
        #     return relationship('Portfolio',
        #                         collection_class=attribute_mapped_collection('ticker'),
        #                         cascade='all, delete-orphan')


class HasStock(object):
    """
    Mixin object to create a relation to a stock object

    Any class that inherits from this class will be given a foreign key column that corresponds to the stock object
    passed to the child class's constructor
    """

    @declared_attr
    def ticker(cls):
        return Column('stock_id', ForeignKey('stock.id'))

        # @declared_attr
        # def stock(cls):
        #     return relationship('Stock')


class Asset(object):
    """
    This is just an empty class acting as a placeholder for my idea that we will later add more than just stock assets
    """
    pass


class Stock(PortfolioAsset, Base):
    """
    main class that is used to model stocks and may contain technical and fundamental data about the stock
    """
    ticker = Column(String, unique=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    get_ohlcv = Column(Boolean)
    load_fundamentals = Column(Boolean)
    beta = Column(Numeric)
    start_price = Column(Numeric)
    end_price = Column(Numeric)
    fundamentals = relationship('Fundamental',
                                collection_class=attribute_mapped_collection('access_key'),
                                cascade='all, delete-orphan')

    def __init__(self, ticker, start_date, end_date, get_fundamentals=False, get_ohlcv=True):
        self.ticker = ticker
        try:
            self.start_date = parser.parse(start_date)
        except ValueError:
            raise ValueError('Error parsing start_date to date. {} was provided'.format(start_date))
        except TypeError:
            # thrown when a datetime is passed in
            self.start_date = start_date

        try:
            self.end_date = parser.parse(end_date)
        except ValueError:
            raise ValueError('could not convert end_date to datetime.datetime. {} was provided'.format(end_date))
        except TypeError:
            # thrown when a datetime is passed in
            self.end_date = end_date

        if self.start_date >= self.end_date:
            raise ValueError(
                'start_date must be older than end_date. start_date: {} end_date: {}'.format(str(start_date),
                                                                                             str(end_date)))
        if self.start_date >= datetime.datetime.now():
            raise ValueError('start_date must be at least older than the current time')

        if self.end_date > datetime.datetime.now():
            raise ValueError('end_date must be at least older than or equal to the current time')

        self.get_ohlcv = get_ohlcv
        if get_ohlcv:
            self.ohlcv = self.get_ohlc_series()
            self.beta = self.calculate_beta()
            self.start_price = self.ohlcv[['Adj Close']].head(1).iloc[0]['Adj Close']
            self.end_price = self.ohlcv[['Adj Close']].tail(1).iloc[0]['Adj Close']
        else:
            self.ohlcv = None
            self.beta = None

        self.fundamentals = {}
        self.load_fundamentals = get_fundamentals
        if get_fundamentals:
            self.get_fundamentals()

    @orm.reconstructor
    def init_on_load(self):
        """
        :return:

        if the user wanted the ohlc_series then recreate it when this object is loaded again
        """
        if self.get_ohlcv:
            self.ohlcv = self.get_ohlc_series()
        else:
            self.ohlcv = None

    # def __getattr__(self, item):
    #     try:
    #         return self.item
    #     except AttributeError:
    #         raise AttributeError(str(item) + ' is not an attribute?')

    def __getitem__(self, key):
        """
        :param key:
        :return:

        I forgot why this is here and am scared to delete it
        """
        return self.ohlcv

    def get_ohlc_series(self, data_source='yahoo'):
        """
        :param data_source: str, see pandas DataReader docs for more valid options. defaults to yahoo
        :return: ohlc pd.Timeseries
        """
        try:
            ohlc = web.DataReader(self.ticker, data_source=data_source, start=self.start_date, end=self.end_date)
            return ohlc
        except:
            logger.exception('Could not create series for ticker: {}. Unknown error occurred.'.format(self.ticker))
            return None

    def get_fundamentals(self):
        """
        :return:

        pass the required attributes to the EdgarSpider and it will create the corresponding fundamentals objects
        for this stock instance as well as write the fundamental object to the DB
        """

        # session = Session()
        with db.query_session() as session:
            # check to see if the fundamentals already exist
            # NOTE: there may be an edge case where the start and end dates are different and so not all of the desired
            # fundamentals will be returned but for now this should work
            # it will require some sort of date guessing based on the periods and the end dates or something so yeah
            # that is a problem for another day
            result = session.query(Fundamental).filter(Fundamental.ticker == self.id).all()
            if result:
                for row in result:
                    self.fundamentals[row.access_key] = row
            else:
                configure_logging()
                runner = CrawlerRunner(settings=get_project_settings())
                spider_dict = {
                    'symbols': self.ticker,
                    'start_date': self.start_date.strftime('%Y%m%d'),
                    'end_date': self.end_date.strftime('%Y%m%d')
                }
                runner.crawl(EdgarSpider, **spider_dict)
                reactor.run()
                result = session.query(Fundamental).filter(Fundamental.ticker == self.id).all()
                for row in result:
                    self.fundamentals[row.access_key] = row

    """
    TECHNICAL INDICATORS/ANALYSIS
    """

    def simple_moving_average(self, period=50, column='Adj Close'):
        return pd.Series(self.ohlcv[column].rolling(center=False, window=period, min_periods=period - 1).mean(),
                         name='{} day SMA Ticker: {}'.format(period, self.ticker)).dropna()

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
        ewma = self._ewma_computation(ts=self.ohlcv, period=period, column=column)
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
        :return: generator

        triangle moving average

        SMA of the SMA
        """
        sma = self._sma_computation(period=period, column=column).rolling(center=False, window=period,
                                                                          min_periods=period - 1).mean()
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
        emwa_one = self._ewma_computation(ts=self.ohlcv, period=period, column=column)
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
        wma = self._weighted_moving_average_computation(ts=self.ohlcv, period=period, column=column)
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
        wma = self._weighted_moving_average_computation(ts=wma_delta, period=sqrt_period, column=column)
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
        :param universe_dict: dict
        :param period_fast: int, traditionally 12
        :param period_slow: int, traditionally 26
        :param signal: int, traditionally 9
        :param column: string
        :return:

        moving average convergence divergence

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
        :param universe_dict: dict
        :param period: int
        :param column: string
        :return: generator

        continually take price differences for a fixed interval

        positive or negative number plotted on a zero line
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
        return pd.Series(self._rsi_computation(ts=self.ohlcv, period=period, column=column),
                         name='{} day RSI Ticker: {}'.format(period, self.ticker))

    def inverse_fisher_transform(self, rsi_period=5, wma_period=9, column='Adj Close'):
        """
        :param universe_dict: dict
        :param rsi_period: int, period that is used for the RSI calculation
        :param wma_period: int, period that is used for the WMA RSI calculation
        :param column: string
        :return: generator

        Modified Inverse Fisher Transform applied on RSI

        Buy when indicator crosses -0.5 or crosses +0.5
        RSI is smoothed with WMA before applying the transformation

        IFT_RSI signals buy when the indicator crosses -0.5 or crosses +0.5 if it has not previously crossed over -0.5
        it signals to sell short when indicators crosses under +0.5 or crosses under -0.5 if it has not previously crossed +.05
        """
        import numpy as np
        v1 = pd.Series(.1 * (self._rsi_computation(ts=self.ohlcv, period=rsi_period, column=column) - 50),
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

        finds the true range a stock is trading within
        most recent period's high - most recent periods low
        absolute value of the most recent period's high minus the previous close
        absolute value of the most recent period's low minus the previous close

        this will give you a dollar amount that the stock's range that it has been trading in
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
        :param universe_dict dict
        :param period: int
        :return: generator

         moving average of a stock's true range
        """
        tr = self._true_range_computation(ts=self.ohlcv, period=period * 2)
        return pd.Series(tr.rolling(center=False, window=period, min_periods=period - 1).mean(),
                         name='{} day ATR Ticker: {}'.format(period, self.ticker)).tail(period)

    def bollinger_bands(self, period=30, moving_average=None, column='Adj Close'):
        std_dev = self.ohlcv[column].std()
        if isinstance(moving_average, pd.Series):
            middle_band = pd.Series(self._sma_computation(ts=self.ohlcv, period=period, column=column),
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

    def _get_pct_change(self, use_portfolio_benchmark=True, market_ticker='^GSPC'):
        """
        Get the percentage change over the :class: Stock's start and end dates for both the stock as well as the market

        :param use_portfolio_benchmark: boolean
            When true the market ticker will be ignored and the ticker set for the whole :class: Portfolio will be used
        :param market_ticker: str
            Any valid ticker symbol to use as the market.
        :return: TimeSeries
        """
        pct_change = namedtuple('Pct_Change', 'market_pct_change stock_pct_change')
        if use_portfolio_benchmark:
            market_df = self._get_portfolio_benchmark()
        else:
            market_df = web.DataReader(market_ticker, 'yahoo', start=self.start_date, end=self.end_date)
        market_pct_change = pd.Series(market_df['Adj Close'].pct_change(periods=1))
        stock_pct_change = pd.Series(self.ohlcv['Adj Close'].pct_change(periods=1))
        return pct_change(market_pct_change=market_pct_change, stock_pct_change=stock_pct_change)

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

    """
    PRIVATE CALCULATION METHODS FOR TECHNICAL INDICATORS/ANALYSIS

    NOTE: these should probably not be class methods but that is something for another day!
    """

    @classmethod
    def _directional_movement_indicator(cls, ts, period):
        """
        :param ts: Series
        :param period: int
        :return: Series

        DMI also known as average directional index
        """
        temp_df = pd.DataFrame()
        temp_df['up_move'] = ts['High'].diff()
        temp_df['down_move'] = ts['Low'].diff()

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
        atr = cls._average_true_range_computation(ts=ts, period=period * 6)
        diplus = pd.Series(100 * (temp_df['positive_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(),
                           name='positive_dmi')
        diminus = pd.Series(100 * (temp_df['negative_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(),
                            name='negative_dmi')
        return pd.concat([diplus, diminus])

    @classmethod
    def _true_range_computation(cls, ts, period):
        """
        :param ts: Timeseries
        :param period: int
        :return: Timeseries

        this method is used internally to compute the average true range of a stock

        the purpose of having it as separate function is so that external functions can return generators
        """
        range_one = pd.Series(ts['High'].tail(period) - ts['Low'].tail(period), name='high_low')
        range_two = pd.Series(ts['High'].tail(period) - ts['Close'].shift(-1).abs().tail(period),
                              name='high_prev_close')
        range_three = pd.Series(ts['Close'].shift(-1).tail(period) - ts['Low'].abs().tail(period),
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

    @classmethod
    def _sma_computation(cls, ts, period=50, column='Adj Close'):
        return pd.Series(ts[column].rolling(center=False, window=period, min_periods=period - 1).mean())

    @classmethod
    def _average_true_range_computation(cls, ts, period):
        tr = cls._true_range_computation(ts, period=period * 2)
        return pd.Series(tr.rolling(center=False, window=period, min_periods=period - 1).mean())

    @classmethod
    def _rsi_computation(cls, ts, period, column):
        """
        :param ts: Series
        :param period: int
        :param column: string
        :return: Series

        relative strength indicator
        """
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

    @classmethod
    def _weighted_moving_average_computation(cls, ts, period, column):
        wma = []
        for chunk in cls._chunks(ts=ts, period=period, column=column):
            # TODO: figure out a better way to handle this. this is better than a catch all except though
            try:
                wma.append(cls.chunked_weighted_moving_average(chunk=chunk, period=period))
            except AttributeError:
                wma.append(None)
        wma.reverse()
        return wma

    @classmethod
    def _chunks(cls, ts, period, column):
        """
        :param ts: Timeseries
        :param period: int, the amount of chunks needed
        :param column: string
        :return: generator

        creates n chunks based on the number of periods
        """
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

    @classmethod
    def _chunked_weighted_moving_average(cls, chunk, period):
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

    @classmethod
    def _ewma_computation(cls, ts, period=50, column='Adj Close'):
        """
        this method is used for computations in other exponential moving averages

        :param ohlc: Timeseries
        :param period: int, number of days
        :param column: string
        :return: Timeseries
        """
        return pd.Series(ts[column].ewm(ignore_na=False, min_periods=period - 1, span=period).mean())

    """
    ALTERNATE CONSTRUCTORS
    """

    @classmethod
    def from_ticker_list(cls, ticker_list, start, end, get_ohlcv=False, get_fundamentals=False):
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
        :param get_fundamentals: boolean
            if true then :class: Fundamentals will be created and added to the db
        :return: generator
            contains one :class: Stock per ticker in the ticker list
        """

        configure_logging()
        runner = CrawlerRunner(settings=get_project_settings())

        spider_dict = {
            'symbols': ticker_list,
            'start_date': start,
            'end_date': end
        }
        runner.crawl(EdgarSpider, **spider_dict)
        d = runner.join()
        d.addBoth(lambda _: reactor.stop())
        reactor.run()

        for ticker in ticker_list:
            yield cls(ticker=ticker, start_date=start, end_date=end, get_ohlcv=get_ohlcv,
                      get_fundamentals=get_fundamentals)

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
        :return: dict, contains stock objects with their corresponding tickers as the key

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

        create a dictionary of stock objects with their tickers as keys and write them to the DB if a session is provided
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


class Fundamental(Base, HasStock):
    """
    the purpose of the this class to hold one period's worth of fundamental data for a given stock

    there should be a column for each argument in the constructor and the allowed list should also be updated when
    new arguments as added and the item_pipeline needs to be updated
    """

    # key to the corresponding Stock's dictionary must be 'period_focus_year'
    access_key = Column(String, unique=True)
    amended = Column(Boolean)
    assets = Column(Numeric(30, 2))
    current_assets = Column(Numeric(30, 2))
    current_liabilities = Column(Numeric(30, 2))
    cash = Column(Numeric(30, 2))
    dividend = Column(Numeric(10, 2))
    end_date = Column(DateTime)
    eps = Column(Numeric(6, 2))
    eps_diluted = Column(Numeric(6, 2))
    equity = Column(Numeric(30, 2))
    net_income = Column(Numeric(30, 2))
    operating_income = Column(Numeric(30, 2))
    revenues = Column(Numeric(30, 2))
    investment_revenues = Column(Numeric(30, 2))
    fin_cash_flow = Column(Numeric(30, 2))
    inv_cash_flow = Column(Numeric(30, 2))
    ops_cash_flow = Column(Numeric(30, 2))
    year = Column(String)
    period_focus = Column(String)
    property_plant_equipment = Column(Numeric(30, 2))
    gross_profit = Column(Numeric(30, 2))
    tax_expense = Column(Numeric(30, 2))
    net_taxes_paid = Column(Numeric(30, 2))
    acts_pay_current = Column(Numeric(30, 2))
    acts_receive_current = Column(Numeric(30, 2))
    acts_receive_noncurrent = Column(Numeric(30, 2))
    acts_receive = Column(Numeric(30, 2))
    accrued_liabilities_current = Column(Numeric(30, 2))
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

    def __init__(self, amended, assets, current_assets, current_liabilities, cash, dividend, end_date, eps, eps_diluted,
                 equity, net_income, operating_income, revenues, investment_revenues, fin_cash_flow, inv_cash_flow,
                 ops_cash_flow, year, property_plant_equipment, gross_profit, tax_expense, net_taxes_paid,
                 acts_pay_current, acts_receive_current, acts_receive_noncurrent, accrued_liabilities_current,
                 period_focus, inventory_net, interest_expense, total_liabilities, total_liabilities_equity,
                 shares_outstanding, shares_outstanding_diluted, common_stock_outstanding, depreciation_amortization,
                 cogs, comprehensive_income_net_of_tax, research_and_dev_expense, warranty_accrual,
                 warranty_accrual_payments):
        """
        This :class: should never really be instantiated directly.  It is intended to be instantiated by the
        :class: EdgarSpider after it has finished scraping the XBRL page that it will be associated with
        because that is where the data will come from that populates the class.

        :param amended: str
            were the finical statements amended
        :param assets: float
            total assets
        :param current_assets: float
            total current assets
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
        # TODO: convert to date. need to test if all dates are the same format
        try:
            self.end_date = parser.parse(end_date)
        except ValueError:
            raise ValueError('end_date could not be converted to datetime object. {} was provided'.format(end_date))
        except TypeError:
            # thrown when a datetime object is passed into the parser
            self.end_date = end_date
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


    def current_ratio(self):
        return self.current_assets / self.current_liabilities

    def return_on_assets(self):
        return self.net_income / self.assets

    def debt_ratio(self):
        return self.assets / self.total_liabilities

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
            raise TypeError('stock must be an instance of a stock object. {} was provided'.format(type(stock)))
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
        # a list of the columns above
        # allowed = ('amended', 'assets', 'current_assets', 'current_liabilities', 'cash', 'dividend', 'end_date', 'eps',
        #            'eps_diluted', 'equity', 'net_income', 'operating_income', 'revenues', 'investment_revenues',
        #            'fin_cash_flow', 'inv_cash_flow', 'ops_cash_flow', 'period_focus', 'year', 'gross_profit',
        #            'property_plant_equipment', 'gross_profit', 'tax_expense', 'net_taxes_paid', 'acts_pay_current',
        #            'acts_receive_current', 'acts_receive_noncurrent', 'accrued_liabilities_current')
        # from scrapy import Selector
        # df = {k : v for k, v in fundamental_dict.items() if k in allowed and type(v) is not Selector}
        df = {k: v for k, v in fundamental_dict.items() if k in cls.__dict__}
        # dd = {}
        # for k, v in fundamental_dict.items():
        #     if k in allowed:
        #         pass
        return cls(**df)

# class FinancialRatios(Fundamental, HasStock, Base):
#     """
#     Holds and calculates all financial ratios and can optionally persist them in the DB
#     """
#     id = Column(Integer, primary_key=True)
#     current_ratio = Column(Numeric(30,6))

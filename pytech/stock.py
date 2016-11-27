import pandas as pd
import os
import numpy as np
import pandas_datareader.data as web
import datetime
import logging
import re
from dateutil import parser
from sqlalchemy import orm

from sqlalchemy.ext.declarative import as_declarative
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import Column, Numeric, String, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship, backref

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

@as_declarative(constructor=None)
class Base(object):

    @declared_attr
    def __tablename__(cls):
        name = cls.__name__
        return (
            name[0].lower() +
            re.sub(r'([A-Z])',
            lambda m: '_' + m.group(0).lower(), name[1:])
        )

    # id = Column(Integer, primary_key=True)


class HasStock(object):
    """
    Mixin object to create a relation to a stock object

    Any class that inherits from this class will be given a foreign key column that corresponds to the stock object
    passed to the child class's constructor
    """

    @declared_attr
    def ticker(cls):
        return Column('ticker', ForeignKey('stock.ticker'))

    @declared_attr
    def stock(cls):
        return relationship('Stock')


class Stock(Base):
    """
    main class that is used to model stocks and may contain technical and fundamental data about the stock
    """
    ticker = Column(String, unique=True, primary_key=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    get_ohlcv = Column(Boolean)
    fundamentals = relationship('Fundamental', backref='fundamentals',cascade='all, delete-orphan')

    def __init__(self, ticker, start_date, end_date, get_fundamentals=False, get_ohlcv=False):
        self.ticker = ticker
        try:
            if type(start_date) == datetime.datetime:
                self.start_date = start_date
            else:
                self.start_date = parser.parse(start_date)
        except ValueError:
            raise ValueError('could not convert start_date to datetime.datetime. {} was provided'.format(start_date))

        try:
            if type(end_date) == datetime.datetime:
                self.end_date = end_date
            else:
                self.end_date = parser.parse(end_date)
        except ValueError:
            raise ValueError('could not convert end_date to datetime.datetime. {} was provided'.format(end_date))

        if self.start_date >= self.end_date:
            raise ValueError('start_date must be older than end_date. start_date: {} end_date: {}'.format(str(start_date),
                                                                                                          str(end_date)))
        if self.start_date >= datetime.datetime.now():
            raise ValueError('start_date must be at least older than the current time')
        if self.end_date > datetime.datetime.now():
            raise ValueError('end_date must be at least older than or equal to the current time')
        self.get_ohlcv = get_ohlcv
        if get_ohlcv:
            self.ohlcv = self.get_ohlc_series()
        else:
            self.ohlcv = None

        self.fundamentals = []
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
        from scrapy.crawler import  CrawlerProcess
        from scrapy.utils.project import get_project_settings
        from crawler.spiders.edgar import EdgarSpider
        from pytech import Session

        session = Session()
        # check to see if the fundamentals already exist
        # NOTE: there may be an edge case where the start and end dates are different and so not all of the desired
        # fundamentals will be returned but for now this should work
        # it will require some sort of date guessing based on the periods and the end dates or something so yeah
        # that is a problem for another day
        fundamentals = session.query(Fundamental).filter(Fundamental.ticker == self.ticker).all()
        if fundamentals:
            self.fundamentals = fundamentals
            session.close()
        else:
            process = CrawlerProcess(get_project_settings())
            spider_dict = {'symbols': self.ticker, 'start_date': self.start_date, 'end_date': self.end_date}
            process.crawl(EdgarSpider, **spider_dict)
            process.start()
            process.join()
            # TODO: test and make sure that after the process finishes the instance has access to ALL of its fundamentals
            self.fundamentals = session.query(Fundamental).filter(Fundamental.ticker == self.ticker).all()
            session.close()

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

    def calculate_beta(self):
        market_df = web.DataReader('SPY', 'yahoo', start=self.start_date, end=self.end_date)
        stock_df = self.ohlcv
        market_start_price = market_df[['Adj Close']].head(1).iloc[0]['Adj Close']
        market_end_price = market_df[['Adj Close']].tail(1).iloc[0]['Adj Close']
        stock_start_price = stock_df[['Adj Close']].head(1).iloc[0]['Adj Close']
        stock_end_price = stock_df[['Adj Close']].tail(1).iloc[0]['Adj Close']
        market_pct_change = pd.Series(market_df['Adj Close'].pct_change(periods=1))
        stock_pct_change = pd.Series(stock_df['Adj Close'].pct_change(periods=1))
        covar = stock_pct_change.cov(market_pct_change)
        variance = market_pct_change.var()
        beta = covar / variance
        correlation = stock_pct_change.corr(market_pct_change)
        market_return = ((market_end_price - market_start_price) / market_start_price) * 100
        stock_return = ((stock_end_price - stock_start_price) / stock_start_price) * 100
        risk_free_rate = web.DataReader('TB1YR', 'fred', start=self.start_date, end=self.end_date).tail(1).iloc[0]['TB1YR']
        market_adj_return = market_return - risk_free_rate
        stock_adj_return = stock_return - risk_free_rate
        return beta

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
        :param ohlc: Timeseries
        :param period: int, number of days
        :param column: string
        :return: Timeseries

        this method is used for computations in other exponential moving averages
        """
        return pd.Series(ts[column].ewm(ignore_na=False, min_periods=period - 1, span=period).mean())

    """
    FUNDAMENTAL ANALYSIS
    """

    def current_ratio(self, full_year=False):
        # TODO: move this to fundamentals
        current_assets = 0.0
        current_liabilities = 0.0
        for fundamental in self.fundamentals:

            if full_year and fundamental.period_focus == 'FY':
                current_assets += fundamental.current_assets
                current_liabilities += fundamental.current_liabilities
            elif fundamental.period_focus != 'FY':
                current_assets += fundamental.current_assets
                current_liabilities += fundamental.current_liabilities
            else:
                 continue
        return current_assets / current_liabilities



    """
    ALTERNATE CONSTRUCTORS
    """

    @classmethod
    def create_stocks_dict_with_fundamentals_from_list(cls, ticker_list, start, end, get_ohlc=False, session=None,
                                                       close_session=True):
        """
        :param ticker_list: list of str objects, they must correspond to a valid ticker symbol
        :param start: datetime, start date
        :param end: datetime, end date
        :param get_ohlc: boolean, if true then a ohlc time series will be created based on the start and end dates
        :param session: sqlalchemy session, if None then one will be created and closed
        :param close_session: boolean, if a user passes a session in and they don't want it to be closed after then set
        this to false
        :return: dict, contains stock objects with their corresponding tickers as the key

        create a dict of stocks for a given time period based on the list of ticker symbols passed in

        corresponding Fundamental objects will also be created and inserted into the db for each stock object created
        """
        from scrapy.crawler import  CrawlerProcess
        from scrapy.utils.project import get_project_settings
        from crawler.spiders.edgar import EdgarSpider
        if session is None:
            from pytech import Session
            session = Session()

        process = CrawlerProcess(get_project_settings())
        for ticker in ticker_list:
            # create a temp dictionary containing the arguments required to create the spider
            temp_dict = {}
            temp_dict['symbols'] = ticker
            temp_dict['start_date'] = start
            temp_dict['end_date'] = end
            process.crawl(EdgarSpider, **temp_dict)
        process.start()
        process.join()

        stock_dict = {}
        for ticker in ticker_list:
            temp_stock = cls(ticker=ticker, start_date=start, end_date=end, get_ohlc=get_ohlc, get_fundamentals=True)
            stock_dict[ticker] = temp_stock
            session.add(temp_stock)
        session.commit()
        if close_session:
            session.close()
        return stock_dict

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
    def create_stocks_dict_from_list(cls, ticker_list, start, end, get_fundamentals=False, get_ohlc=False, session=None):
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
    """

    id = Column(Integer, primary_key=True)
    amended = Column(Boolean)
    assets = Column(Numeric(30, 2))
    current_assets = Column(Numeric(30, 2))
    current_liabilities = Column(Numeric(30, 2))
    cash = Column(Numeric(30, 2))
    dividend = Column(Numeric(10, 2))
    end_date = Column(DateTime)
    eps = Column(Numeric(6,2))
    eps_diluted  = Column(Numeric(6,2))
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
    # period_year = Column(String)


    def __init__(self, amended, assets, current_assets, current_liabilities, cash, dividend, end_date, eps, eps_diluted,
                 equity, net_income, operating_income, revenues, investment_revenues, fin_cash_flow, inv_cash_flow, ops_cash_flow,
                 ticker, year, period_focus=None):
        """
        :param stock:
        :param year:
        :param period_focus: defaults to full year. other valid input is Q1, Q2, or Q3
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
        self.ticker = ticker
        # self.period_year = period_year

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
        return cls(amended=amended, assets=assets, current_assets=current_assets, current_liabilities=current_liabilities,
                   cash=cash, dividend=dividend, end_date=end_date, eps=eps, eps_diluted=eps_diluted, equity=equity,
                   net_income=net_income, operating_income=operating_income, revenues=revenues,
                   investment_revenues=investment_revenues, fin_cash_flow=fin_cash_flow, inv_cash_flow=inv_cash_flow,
                   ops_cash_flow=ops_cash_flow, year=year, period_focus=period_focus, ticker=ticker)

    @classmethod
    def from_dict(cls, fundamental_dict):
        # a list of the columns above
        allowed = ('amended', 'assets', 'current_assets', 'current_liabilities', 'cash', 'dividend', 'end_date', 'eps',
                   'eps_diluted', 'equity', 'net_income', 'operating_income', 'revenues', 'investment_revenues',
                   'fin_cash_flow', 'inv_cash_flow', 'ops_cash_flow', 'period_focus', 'year', 'ticker')
        df = {k : v for k, v in fundamental_dict.items() if k in allowed}
        return cls(**df)





class Trade(HasStock, Base):
    id = Column(Integer, primary_key=True)
    trade_date = Column(DateTime)
    # buy or sell
    action = Column(String)
    # long or short
    position = Column(String)
    qty = Column(Integer)
    price_per_share = Column(Numeric(9,2))
    corresponding_trade_id = Column(Integer, ForeignKey('trade.id'))
    corresponding_trade = relationship('Trade', remote_side=[id])

    def __init__(self, trade_date, qty, price_per_share, stock, action='buy', position=None, corresponding_trade=None):
        """
        :param trade_date: datetime.datetime, corresponding to the trade date
        :param qty: int, number of shares traded
        :param price_per_share: float, price per individual share in the trade or the average share price in the trade
        :param stock: Stock, the stock object that was traded
        :param action: str, buy or sell depending on what kind of trade it was
        :param position: str, long or short
        """
        if type(trade_date) is datetime.datetime:
            self.trade_date = trade_date
        else:
            raise TypeError('trade_date must be of type datetime.datetime. '
                            '{} is not a datetime.datetime object'.format(trade_date))
        if action.lower() == 'buy' or action.lower() == 'sell':
            # TODO: may have to run a query to check if we own the stock or not? and if we do use update?
            self.action = action.lower()
        else:
            raise ValueError('action must be either "buy" or "sell". {} was provided.'.format(action))
        if position.lower() == 'long' or position.lower() == 'short':
            self.position = position.lower()
        elif position is None and corresponding_trade is not None:
            self.position = position
        elif position is None and corresponding_trade is None:
            raise ValueError('position can only be None if a corresponding_trade is also provided and None was provided')
        else:
            raise ValueError('position must be either "long" or "short". {} was provided.'.format(position))
        if isinstance(stock, Stock):
            self.stock = stock
        else:
            raise ValueError('stock must be an instance of the Stock class. {} was provided.'.format(stock))
        if corresponding_trade is None or isinstance(corresponding_trade, Trade):
            # TODO: check if the corresponding_trade is actually in the DB yet
            self.corresponding_trade = corresponding_trade
        else:
            raise ValueError('corresponding_trade must either be None or an instance of a Trade object')
        # TODO: if the position is short shouldn't this be negative?
        self.qty = qty
        self.price_per_share = price_per_share

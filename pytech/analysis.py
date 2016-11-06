import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from matplotlib.finance import candlestick_ohlc
from pytech.portfolio import Portfolio
import sys

import finsymbols

sys.path.append('/home/kp/CodeFiles/StockPicker/')


def main():
    portfolio = Portfolio(tickers=['AAPL', 'SPY'])
    universe_dict = portfolio.asset_dict
    # universe_dict = get_stock_universe(stock_list=['AAPL', 'SPY'])
    for i in simple_moving_average(universe_dict):
        print(i.tail())
    for i in simple_moving_median(universe_dict):
        print(i.tail())
    for i in exponential_weighted_moving_average(universe_dict):
        print(i.tail())
    for i in double_ewma(universe_dict):
        print(i.tail())
    for i in triple_ewma(universe_dict):
        print(i.tail())
    for i in triangle_moving_average(universe_dict):
        print(i.tail())
    for i in triple_ema_oscillator(universe_dict):
        print(i.tail())
    for i in efficiency_ratio(universe_dict):
        print(i.tail())
    for i in kama(universe_dict):
        print(i.tail())
    for i in zero_lag_ema(universe_dict):
        print(i.tail())
    for i in weighted_moving_average(universe_dict):
        print(i.tail())
    for i in hull_moving_average(universe_dict):
        print(i.tail())
    for i in smoothed_moving_average(universe_dict):
        print(i.tail())
    for i in macd_signal(universe_dict):
        print(i.tail())
    for i in market_momentum(universe_dict):
        print(i.tail())
    for i in rate_of_change(universe_dict):
        print(i.tail())
    for i in relative_strength_indicator(universe_dict):
        print(i.tail())
    for i in inverse_fisher_transform(universe_dict):
        print(i.tail())
    for i in true_range(universe_dict):
        print(i.tail())
    for i in average_true_range(universe_dict):
        print(i.tail())
    _get_stock_beta(universe_dict, 'AAPL')
    # print(pd.read_csv("http://finance.yahoo.com/d/quotes.csv?s=AAPL+GOOG+MSFT&f=nabe"))


default_start = datetime.datetime.today() - datetime.timedelta(days=365)
default_end = datetime.date.today()


def test(start=default_start, end=default_end):
    print(web.DataReader('TB1YR', 'fred', start=start, end=end))


def get_stock_universe(start=default_start, end=default_end, **kwargs):
    """
    :param start: datetime.datetime, default t - 1 year
    :param end: datetime.datetime, default t
    :param kwargs: stock_list = list, list of ticker symbols that you wish to trade
    :return: dict, key=ticker value=timeseries containing the day over day ohlc for the stocks in your universe

    this method creates the data that will used for all trading decisions. only stocks in this series will be considered
    for trading decisions.
    """
    df_list = []
    df_dict = {}
    if 'stock_list' in kwargs:
        stock_list = kwargs['stock_list']
        for stock in stock_list:
            # df_list.append(web.DataReader(stock, data_source="yahoo", start=start,
            #                               end=end))
            temp_df = web.DataReader(stock, data_source='yahoo', start=start, end=end)
            temp_df['ticker'] = stock
            df_dict[stock] = temp_df
            df_list.append(temp_df)
    else:
        stock_list = []
        sp = finsymbols.get_sp500_symbols()
        for dict in sp:
            symbol = dict.get("symbol")
            # df_list.append(web.DataReader(symbol, data_source="yahoo", start=start, end=end))
            temp_df = web.DataReader(symbol, data_source='yahoo', start=start, end=end)
            temp_df['ticker'] = symbol
            df_dict[symbol] = temp_df
            df_list.append(temp_df)
    return df_dict


def simple_moving_average(universe_dict, period=50, column='Adj Close'):
    """
    :param ohlc: dict
    :param period: int, the number of days to use
    :param column: string, the name of the column to use to compute the mean
    :return: Timeseries containing the simple moving average

    compute the simple moving average over a given period and return it in timeseries
    """
    for ticker, ts in universe_dict.items():
        # temp_ts = pd.Series(ts[column].rolling(center=False, window=period, min_periods=period - 1).mean(),
        #                     name='{} day SMA Ticker: {}'.format(period, ticker))
        # yield temp_ts
        yield pd.Series(ts[column].rolling(center=False, window=period, min_periods=period - 1).mean(),
                        name='{} day SMA Ticker: {}'.format(period, ticker))


def _sma_computation(ohlc, period=50, column='Adj Close'):
    return pd.Series(ohlc[column].rolling(center=False, window=period, min_periods=period - 1).mean())


def simple_moving_median(universe_dict, period=50, column='Adj Close'):
    """
    :param ohlc: dict
    :param period: int, the number of days to use
    :param column: string, the name of the column to use to compute the median
    :return: Timeseries containing the simple moving median

    compute the simple moving median over a given period and return it in timeseries
    """
    for ticker, ts in universe_dict.items():
        yield pd.Series(ts[column].rolling(center=False, window=period, min_periods=period - 1).median(),
                        name='{} day SMM Ticker: {}'.format(period, ticker))


def exponential_weighted_moving_average(universe_dict, period=50, column='Adj Close'):
    """
    :param ohlc: dict
    :param period: int, the number of days to use
    :param column: string, the name of the column to use to compute the mean
    :return: Timeseries containing the simple moving median

    compute the exponential weighted moving average (ewma) over a given period and return it in timeseries
    """
    for ticker, ts in universe_dict.items():
        yield pd.Series(ts[column].ewm(ignore_na=False, min_periods=period - 1, span=period).mean(),
                        name='{} day EWMA Ticker: {}'.format(period, ticker))


def _ewma_computation(ohlc: pd.Series, period: int = 50, column: str = 'Adj Close') -> pd.Series:
    """
    :param ohlc: Timeseries
    :param period: int, number of days
    :param column: string
    :return: Timeseries

    this method is used for computations in other exponential moving averages
    """
    return pd.Series(ohlc[column].ewm(ignore_na=False, min_periods=period - 1, span=period).mean())


def double_ewma(universe_dict, period=50, column='Adj Close'):
    """

    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    double exponential moving average
    """
    for ticker, ts in universe_dict.items():
        ewma = _ewma_computation(ohlc=ts, period=period, column=column)
        ewma_mean = ewma.ewm(ignore_na=False, min_periods=period -1, span=period).mean()
        dema = 2 * ewma - ewma_mean
        yield pd.Series(dema, name='{} day DEMA Ticker: {}'.format(period, ticker))


def triple_ewma(universe_dict, period=50, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    triple exponential moving average
    """
    for ticker, ts in universe_dict.items():
        ewma = _ewma_computation(ohlc=ts, period=period, column=column)
        triple_ema = 3 * ewma
        ema_ema_ema = ewma.ewm(ignore_na=False, span=period).mean().ewm(ignore_na=False, span=period).mean()
        tema = triple_ema - 3 * ewma.ewm(ignore_na=False, min_periods=period - 1, span=period).mean() + ema_ema_ema
        yield pd.Series(tema, name='{} day TEMA Ticker: {}'.format(period, ticker))


def triangle_moving_average(universe_dict, period=50, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    triangle moving average

    SMA of the SMA
    """
    for ticker, ts in universe_dict.items():
        sma = _sma_computation(ohlc=ts, period=period, column=column).rolling(center=False, window=period,
                                                                              min_periods=period - 1).mean()
        yield pd.Series(sma, name='{} day TRIMA Ticker: {}'.format(period, ticker))


def triple_ema_oscillator(universe_dict, period=15, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    triple exponential moving average oscillator (trix)

    calculates the triple smoothed EMA of n periods and finds the pct change between 1 period of EMA3

    oscillates around 0. positive numbers indicate a bullish indicator
    """
    for ticker, ts in universe_dict.items():
        emwa_one = _ewma_computation(ohlc=ts, period=period, column=column)
        emwa_two = emwa_one.ewm(ignore_na=False, min_periods=period - 1, span=period).mean()
        emwa_three = emwa_two.ewm(ignore_na=False, min_periods=period - 1, span=period).mean()
        trix = emwa_three.pct_change(periods=1)
        yield pd.Series(trix, name='{} days TRIX Ticker: {}'.format(period, ticker))


def efficiency_ratio(universe_dict, period=10, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    Kaufman Efficiency Indicator. oscillates between +100 and -100

    positive is bullish
    """
    for ticker, ts in universe_dict.items():
        change = ts[column].diff(periods=period).abs()
        vol = ts[column].diff().abs().rolling(window=period).sum()
        yield pd.Series(change / vol, name='{} days Efficiency Indicator Ticker: {}'.format(period, ticker))


def _efficiency_ratio_computation(ohlc, period=10, column='Adj Close'):
    """
    :param ohlc: Timeseries
    :param period: int, days
    :param column: string
    :return: Timeseries

    Kaufman Efficiency Indicator. oscillates between +100 and -100

    positive is bullish
    """

    change = ohlc[column].diff(periods=period).abs()
    vol = ohlc[column].diff().abs().rolling(window=period).sum()
    return pd.Series(change / vol)


def kama(universe_dict, efficiency_ratio_periods=10, ema_fast=2, ema_slow=30, period=20, column='Adj Close'):
    for ticker, ts in universe_dict.items():
        er = _efficiency_ratio_computation(ohlc=ts, period=efficiency_ratio_periods, column=column)
        fast_alpha = 2 / (ema_fast + 1)
        slow_alpha = 2 / (ema_slow + 1)
        smoothing_constant = pd.Series((er * (fast_alpha - slow_alpha) + slow_alpha) ** 2, name='smoothing_constant')
        sma = pd.Series(ts[column].rolling(period).mean(), name='SMA')
        kama = []
        for smooth, ma, price in zip(iter(smoothing_constant.items()), iter(sma.shift(-1).items()), iter(ts[column].items())):
            try:
                kama.append(kama[-1] + smooth[1] * (price[1] - kama[-1]))
            except:
                if pd.notnull(ma[1]):
                    kama.append(ma[1] + smooth[1] * (price[1] - ma[1]))
                else:
                    kama.append(None)
        sma['KAMA'] = pd.Series(kama, index=sma.index, name='{} days KAMA Ticker {}'.format(period, ticker))
        yield sma['KAMA']


def zero_lag_ema(universe_dict, period=30, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    zero lag exponential moving average

    """
    lag = (period - 1) / 2
    for ticker, ts in universe_dict.items():
        yield pd.Series((ts[column] + (ts[column].diff(lag))), name='{} days Zero Lag EMA Ticker: {}'.format(period, ticker))


def weighted_moving_average(universe_dict, period=30, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    aims to smooth the price curve for better trend identification
    places a higher importance on recent data compared to the EMA
    """
    for ticker, ts in universe_dict.items():
        wma = _weighted_moving_average_computation(ts=ts, period=period, column=column)
        # ts['WMA'] = pd.Series(wma, index=ts.index)
        yield pd.Series(pd.Series(wma, index=ts.index), name='{} days WMA Ticker: {}'.format(period, ticker))
        # yield pd.Series(ts['WMA'], name='{} days WMA Ticker: {}'.format(period, ticker))


def hull_moving_average(universe_dict, period=30, column='Adj Close'):
    """

    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    smoother than the SMA, it aims to minimize lag and track price trends more accurately

    best used in mid to long term analysis
    """
    import math
    for ticker, ts in universe_dict.items():
        wma_one_period = int(period / 2) * 2
        wma_one = pd.Series(_weighted_moving_average_computation(ts=ts, period=wma_one_period, column=column),
                            index=ts.index)
        wma_one *= 2
        wma_two = pd.Series(_weighted_moving_average_computation(ts=ts, period=period, column=column), index=ts.index)
        wma_delta = wma_one - wma_two
        sqrt_period = int(math.sqrt(period))
        wma = _weighted_moving_average_computation(ts=wma_delta, period=sqrt_period, column=column)
        wma_delta['_WMA'] = pd.Series(wma, index=ts.index)
        yield pd.Series(wma_delta['_WMA'], name='{} day HMA Ticker: {}'.format(period, ticker))


def volume_weighted_moving_average(universe_dict, period=30, column='Adj Close'):
    pass


def smoothed_moving_average(universe_dict, period=30, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int, days
    :param column: string
    :return: generator

    equal weights given to historic and more current prices
    """
    for ticker, ts in universe_dict.items():
        yield pd.Series(ts[column].ewm(alpha=1 / float(period)).mean(),
                        name='{} days SMMA Ticker: {}'.format(period, ticker))


def macd_signal(universe_dict, period_fast=12, period_slow=26, signal=9, column='Adj Close'):
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
    for ticker, ts in universe_dict.items():
        ema_fast = pd.Series(ts[column].ewm(ignore_na=False, min_periods=period_fast - 1, span=period_fast).mean(),
                             name='EMA_fast')
        ema_slow = pd.Series(ts[column].ewm(ignore_na=False, min_periods=period_slow - 1, span=period_slow).mean(),
                             name='EMA_slow')
        macd_series = pd.Series(ema_fast - ema_slow, name='MACD')
        macd_signal_series = pd.Series(macd_series.ewm(ignore_na=False, span=signal).mean(), name='MACD_Signal')
        yield pd.concat([macd_signal_series, macd_series], axis=1)


def market_momentum(universe_dict, period=10, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int
    :param column: string
    :return: generator

    continually take price differences for a fixed interval

    positive or negative number plotted on a zero line
    """
    for ticker, ts in universe_dict.items():
        yield pd.Series(ts[column].diff(period), name='{} day MOM Ticker: {}'.format(period, ticker))


def rate_of_change(universe_dict, period=1, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int
    :param column: string
    :return: generator

    simply calculates the rate of change between two periods
    """
    for ticker, ts in universe_dict.items():
        yield pd.Series((ts[column].diff(period) / ts[column][-period]) * 100,
                        name='{} day Rate of Change Ticker: {}'.format(period, ticker))


def relative_strength_indicator(universe_dict, period=14, column='Adj Close'):
    """
    :param universe_dict: dict
    :param period: int
    :param column: string
    :return: generator

    RSI oscillates between 0 and 100 and traditionally +70 is considered overbought and under 30 is oversold
    """
    for ticker, ts in universe_dict.items():
        print(ts)
        yield pd.Series(_rsi_computation(ts=ts, period=period, column=column),
                        name='{} day RSI Ticker: {}'.format(period, ticker))


def inverse_fisher_transform(universe_dict, rsi_period=5, wma_period=9, column='Adj Close'):
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
    for ticker, ts in universe_dict.items():
        v1 = pd.Series(.1 * (_rsi_computation(ts=ts, period=rsi_period, column=column) - 50),
                       name='v1')
        v2 = pd.Series(_weighted_moving_average_computation(ts=v1, period=wma_period, column=column), index=v1.index)
        yield pd.Series((np.exp(2 * v2) - 1) / (np.exp(2 * v2) + 1),
                        name='{} day IFT_RSI Ticker: {}'.format(rsi_period, ticker))


def true_range(universe_dict, period=14):
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
    for ticker, ts in universe_dict.items():
        # TODO: make this method use adjusted close
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
        yield pd.Series(tr['TA'], name='{} day TR Ticker: {}'.format(period, ticker))


def average_true_range(universe_dict, period=14):
    """
    :param universe_dict dict
    :param period: int
    :return: generator

     moving average of a stock's true range
    """
    for ticker, ts in universe_dict.items():
        tr = _true_range_computation(ts, period=period * 2)
        yield pd.Series(tr.rolling(center=False, window=period, min_periods=period - 1).mean(),
                        name='{} day ATR Ticker: {}'.format(period, ticker)).tail(period)


def bollinger_bands(universe_dict, period=30, moving_average=None, column='Adj Close'):
    for ticker, ts in universe_dict.items():
        std_dev = ts[column].std()
        if isinstance(moving_average, pd.Series):
            middle_band = pd.Series(_sma_computation(ts, period=period, column=column), name='middle_bband')
        else:
            middle_band = pd.Series(moving_average, name='middle_bband')

        upper_bband = pd.Series(middle_band + (2 * std_dev), name='upper_bband')
        lower_bband = pd.Series(middle_band - (2 * std_dev), name='lower_bband')

        percent_b = pd.Series((ts[column] - lower_bband) / (upper_bband - lower_bband), name='%b')
        b_bandwidth = pd.Series((upper_bband - lower_bband) / middle_band, name='b_bandwidth')
        yield pd.concat([upper_bband, middle_band, lower_bband, b_bandwidth, percent_b], axis=1)


def _get_stock_beta(universe_dict, ticker):
    if 'SPY' in universe_dict:
        market_df = universe_dict['SPY']
    else:
        market_df = web.DataReader('SPY', 'yahoo', start=default_start, end=default_end)
    stock_df = universe_dict[ticker]
    market_start_price = market_df[['Adj Close']].head(1).iloc[0]['Adj Close']
    market_end_price = market_df[['Adj Close']].tail(1).iloc[0]['Adj Close']
    stock_start_price = stock_df[['Adj Close']].head(1).iloc[0]['Adj Close']
    stock_end_price = stock_df[['Adj Close']].tail(1).iloc[0]['Adj Close']
    market_pct_change = pd.Series(market_df['Adj Close'].pct_change(periods=1))
    stock_pct_change = pd.Series(stock_df['Adj Close'].pct_change(periods=1))
    covar = market_pct_change.cov(stock_pct_change)
    print(covar)
    covar = stock_pct_change.cov(market_pct_change)
    print(covar)
    variance = market_pct_change.var()
    print(variance)
    beta = covar / variance
    correlation = stock_pct_change.corr(market_pct_change)
    print(correlation)
    market_return = ((market_end_price - market_start_price) / market_start_price) * 100
    stock_return = ((stock_end_price - stock_start_price) / stock_start_price) * 100
    risk_free_rate = web.DataReader('TB1YR', 'fred', start=default_start, end=default_end).tail(1).iloc[0]['TB1YR']
    market_adj_return = market_return - risk_free_rate
    stock_adj_return = stock_return - risk_free_rate
    print(beta)


def directional_movement_indicator(universe_dict, period=14):
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
    for ticker, ts in universe_dict.items():
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
        atr = _average_true_range_computation(ts=ts, period=period * 6)
        diplus = pd.Series(100 * (temp_df['positive_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(), name='positive_dmi')
        diminus = pd.Series(100 * (temp_df['negative_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(), name='negative_dmi')
        yield pd.concat([diplus, diminus])


def _true_range_computation(ts, period):
    """
    :param ts: Timeseries
    :param period: int
    :return: Timeseries

    this method is used internally to compute the average true range of a stock

    the purpose of having it as separate function is so that external functions can return generators
    """
    range_one = pd.Series(ts['High'].tail(period) - ts['Low'].tail(period), name='high_low')
    range_two = pd.Series(ts['High'].tail(period) - ts['Close'].shift(-1).abs().tail(period), name='high_prev_close')
    range_three = pd.Series(ts['Close'].shift(-1).tail(period) - ts['Low'].abs().tail(period), name='prev_close_low')
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


def _average_true_range_computation(ts, period):
    tr = _true_range_computation(ts, period=period * 2)
    return pd.Series(tr.rolling(center=False, window=period, min_periods=period - 1).mean())


def _directional_movement_indicator(ts, period):
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
    atr = _average_true_range_computation(ts=ts, period=period * 6)
    diplus = pd.Series(100 * (temp_df['positive_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(), name='positive_dmi')
    # diplus = pd.Series( 100 * (temp_df['positive_dm'] / _average_true_range_computation(ts=ts, period=period * 6)).ewm(span=period, min_periods=period - 1).mean(), name='positive_dmi')
    diminus = pd.Series(100 * (temp_df['negative_dm'] / atr).ewm(span=period, min_periods=period - 1).mean(), name='negative_dmi')
    # diminus = pd.Series( 100 * (temp_df['negative_dm'] / _average_true_range_computation(ts=ts, period=period * 6)).ewm(span=period, min_periods=period - 1).mean(), name='negative_dmi')
    yield pd.concat([diplus, diminus])

def _rsi_computation(ts, period, column):
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


def _weighted_moving_average_computation(ts, period, column):
    wma = []
    for chunk in _chunks(ts=ts, period=period, column=column):
        # TODO: figure out a better way to handle this. this is better than a catch all except though
        try:
            wma.append(_chunked_weighted_moving_average(chunk=chunk, period=period))
        except AttributeError:
            wma.append(None)
    wma.reverse()
    return wma


def _chunks(ts, period, column):
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


if __name__ == "__main__":
    main()

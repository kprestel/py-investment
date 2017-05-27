"""
Contains functions to perform technical analysis on pandas OHLCV data frames
"""
import pandas as pd

import pytech.utils.pandas_utils as pd_utils
from pytech.utils.decorators import memoize


@memoize
def sma(df: pd.DataFrame,
        period: int = 50,
        col: str = pd_utils.ADJ_CLOSE_COL) -> pd.Series:
    """
    Simple moving average

    :param df: The data frame to perform the sma on.
    :param period: The length of the moving average
    :param col: The column in the data frame to use.
    :return: A series with the simple moving average
    """
    sma = df[col].rolling(center=False, min_periods=period - 1).mean()
    return sma.dropna()


def simple_moving_median(df: pd.DataFrame,
                         period: int = 50,
                         column: str = pd_utils.ADJ_CLOSE_COL) -> pd.Series:
    """
    Compute the simple moving median over a given period.

    :param df: The data frame.
    :param period: The number of days to use.
    :param column: The name of the column to use to compute the median.
    :return: Series containing the simple moving median.

    """
    return df[column].rolling(center=False,
                              window=period,
                              min_periods=period - 1).median()


def ewma(df: pd.DataFrame, period: int = 50,
         col: str = pd_utils.ADJ_CLOSE_COL) -> pd.Series:
    """
    Exponential weighted moving average.

    :param df:
    :param period:
    :param col:
    :return:
    """
    return df[col].ewm(ignore_na=False,
                       min_periods=period - 1,
                       span=period).mean()


# noinspection PyTypeChecker
def triple_ewma(df: pd.DataFrame, period: int = 50,
                col: str = pd_utils.ADJ_CLOSE_COL) -> pd.Series:
    """
    Triple Exponential Weighted Moving Average.

    :param df: The data frame to preform the calculation on.
    :param period: The number of periods.
    :param col: The column to perform the calculation on.
    :return:
    """
    ewma_ = ewma(df, period, col)
    triple_ema = 3 * ewma_
    ema_ema_ema = (ewma_.ewm(ignore_na=False, span=period).mean()
                   .ewm(ignore_na=False, span=period).mean())
    return triple_ema - 3 * (ewma_.ewm(ignore_na=False,
                                       min_periods=period - 1,
                                       span=period).mean()) + ema_ema_ema


def triangle_ma(df: pd.DataFrame, period: int = 50,
                col: str = pd_utils.ADJ_CLOSE_COL) -> pd.Series:
    """
    Triangle Moving Average. The SMA of the SMA.

    :param df: The data frame to preform the calculation on.
    :param period: The number of periods.
    :param col: The column to use to do the calculation.
    :return:
    """
    sma_ = sma(df, period, col)
    return sma_.rolling(center=False, window=period,
                        min_periods=period - 1).mean


def trix(df: pd.DataFrame, period: int = 50,
         col: str = pd_utils.ADJ_CLOSE_COL) -> pd.Series:
    """
    Triple Exponential Moving Average Oscillator (trix)

    Calculates the tripe EMA of `n` periods and finds the percent change
    between 1 period of EMA3

    Oscillates around 0. positive numbers indicate a bullish indicator.

    :param df: The data frame to preform the calculation on.
    :param period: The number of periods.
    :param col: The column to use to do the calculation.
    :return:
    """
    emwa_one = ewma(df, period, col)
    emwa_two = emwa_one.ewm(ignore_na=False,
                            min_periods=period - 1,
                            span=period).mean()
    emwa_three = emwa_two.ewm(ignore_na=False,
                            min_periods=period - 1,
                            span=period).mean()

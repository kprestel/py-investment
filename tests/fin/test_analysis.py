import pytest
import logging
import pandas as pd

import pytech.fin.tech_analysis as ta
import pytech.utils.pandas_utils as pd_utils

logger = logging.getLogger(__name__)


# TODO: better assertions


def test_sma(aapl_df):
    """Test of the SMA."""
    logger.info(aapl_df.index)
    df = ta.sma(aapl_df)
    assert isinstance(df, pd.Series)
    high_df = ta.sma(aapl_df, col=pd_utils.HIGH_COL)
    assert not df.equals(high_df)


def test_smm(aapl_df):
    """Test of the SMM"""
    df = ta.smm(aapl_df)
    assert isinstance(df, pd.Series)
    logger.debug(df)
    high_df = ta.smm(aapl_df, col=pd_utils.HIGH_COL)
    assert not df.equals(high_df)


def test_ewma(aapl_df):
    """Test of the EWMA"""
    df = ta.ewma(aapl_df)
    assert isinstance(df, pd.Series)
    logger.debug(df)
    high_df = ta.ewma(aapl_df, col=pd_utils.HIGH_COL)
    assert not df.equals(high_df)


def test_triple_ewma(aapl_df):
    df = ta.triple_ewma(aapl_df)
    assert isinstance(df, pd.Series)
    logger.debug(df)
    high_df = ta.triple_ewma(aapl_df, col=pd_utils.HIGH_COL)
    assert not df.equals(high_df)


def test_triangle_ma(aapl_df):
    """Test the triangle MA."""
    df = ta.triangle_ma(aapl_df)
    assert isinstance(df, pd.Series)
    logger.debug(df)
    high_df = ta.triangle_ma(aapl_df, col=pd_utils.HIGH_COL)
    assert not df.equals(high_df)


def test_trix(aapl_df):
    """Test the TRIX."""
    df = ta.trix(aapl_df)
    assert isinstance(df, pd.Series)
    high_df = ta.trix(aapl_df, col=pd_utils.HIGH_COL)
    assert not df.equals(high_df)


def test_efficiency_ratio(aapl_df):
    """Test the ER."""
    df = ta.efficiency_ratio(aapl_df)
    assert isinstance(df, pd.Series)
    high_df = ta.efficiency_ratio(aapl_df, col=pd_utils.HIGH_COL)
    assert not df.equals(high_df)


def test_kama(aapl_df):
    """Test the KAMA."""
    df = ta.kama(aapl_df)
    assert isinstance(df, pd.Series)
    high_df = ta.kama(aapl_df, col=pd_utils.HIGH_COL)
    logger.debug(df)
    assert not df.equals(high_df)


def test_zero_lag_ema(aapl_df):
    """Test Zero Lag EMA"""
    df = ta.zero_lag_ema(aapl_df)
    assert isinstance(df, pd.Series)
    high_df = ta.zero_lag_ema(aapl_df, col=pd_utils.HIGH_COL)
    logger.debug(df)
    assert not df.equals(high_df)

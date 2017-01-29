from contextlib import contextmanager
import numpy as np
from functools import wraps
from pytech import Session, engine
import pandas as pd

def ohlcv_to_sql(asset):
    """
    Write a `pandas.DataFrame` to the DB.

    :param Asset asset: An instance of a `:py:class:pytech.asset.Asset`
    :return: None, this method only writes to the DB.
    """

    df = pd.DataFrame(asset.ohlcv)
    df['asof_date'] = _dt_to_epoch(df['asof_date'])
    df['ticker'] = asset.ticker
    df.to_sql('universe_ohlcv', con=engine, if_exists='append', index=False)

def _dt_to_epoch(dt_series):
    """
    Convert a ``pandas.TimeSeries`` to epoch time stamps.

    This method is inspired by the zipline library.

    :param TimeSeries dt_series: The series of dates to convert.
    :return: The dt_series converted to epoch timestamps in nanoseconds.
    """

    index = pd.to_datetime(dt_series.values)

    if index.tz_info is None:
        index = index.tz_localize('UTC')
    else:
        index - index.tz_convert('UTC')
    return index.view(np.int64)


def ohlcv_from_sql(asset=None, date_to_load=None):
    """
    Retrieve a `pandas.DataFrame` from the DB.

    :param asset: The asset to retrieve the OHLCV of. If left None then all of the OHLCVs in the DB will be loaded.
    (default: None)
    :type asset: :py:class:`pytech.asset.Asset` or None
    :param datetime date_to_load: Only retrieve the OHLCV data for a particular date.
        Used for things such as backtesting.
    :return: The OHLCV `pandas.DataFrame`
    :rtype: `pandas.DataFrame`

    .. note::
        *

    """
    # TODO: add date ranges in kwargs.
    # TODO: add option to load only particular cols.

    if asset is None and date_to_load is None:
        sql = 'SELECT * FROM ohlcv'
        params = {}
    elif asset is None and date_to_load is not None:
        sql = 'SELECT * FROM ohlcv WHERE trade_date = :trade_date'
    elif asset is not None and date_to_load is None:
        sql = 'SELECT * FROM ohlcv WHERE ticker = :ticker'
    elif asset is not None and date_to_load is not None:
        sql = 'SELECT * FROM ohlcv WHERE ticker = :ticker AND trade_date = :trade_date'
    else:
        raise ValueError('Invalid Arguments')





# @contextmanager
def raw_connection(*args, **kwargs):
    return engine


@contextmanager
def query_session():
    session = Session()
    yield session
    session.close()

@contextmanager
def transactional_session(auto_close=False):
    """
    Helper method to manage db transactions

    :param auto_close: whether or not the session should be closed after it goes out of scope
    :type auto_close: bool
    :return: session
    """

    session = Session()
    # session.begin(nested=nested)
    try:
        yield session
    except:
        session.rollback()
        raise
    else:
        session.commit()
        if auto_close:
            session.close()

def in_transaction(**session_kwargs):
    """Decorator which wraps the decorated function in a transactional session. If the
       function completes successfully, the transaction is committed. If not, the transaction
       is rolled back."""
    def outer_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with transactional_session(**session_kwargs) as session:
                return func(session, *args, **kwargs)
        return wrapper
    return outer_wrapper
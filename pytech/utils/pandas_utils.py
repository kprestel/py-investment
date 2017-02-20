import pandas as pd

def rename_yahoo_ohlcv_cols(df):
    """
    Rename the default return columns from Yahoo to the format that the DB expects.

    :param DataFrame df: The ``DataFrame`` that needs the columns renamed.
    :return:
    """

    return df.rename(columns={
        'Date': 'asof_date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    })

import pandas as pd

# constants for the expected column names of ALL ohlcv DataFrames

DATE_COL = 'asof_date'
OPEN_COL = 'open'
HIGH_COL = 'high'
LOW_COL = 'low'
CLOSE_COL = 'close'
ADJ_CLOSE_COL = 'adj_close'
VOL_COL = 'volume'


def rename_yahoo_ohlcv_cols(df):
    """
    Rename the default return columns from Yahoo to the format that the DB expects.

    :param DataFrame df: The ``DataFrame`` that needs the columns renamed.
    :return:
    """

    return df.rename(columns={
        'Date': DATE_COL,
        'Open': OPEN_COL,
        'High': HIGH_COL,
        'Low': LOW_COL,
        'Close': CLOSE_COL,
        'Adj Close': ADJ_CLOSE_COL,
        'Volume': VOL_COL
    })

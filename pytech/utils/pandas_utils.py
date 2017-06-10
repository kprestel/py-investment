import numpy as np
import pandas as pd
import xarray as xr

# constants for the expected column names of ALL data DataFrames

DATE_COL = 'date'
OPEN_COL = 'open'
HIGH_COL = 'high'
LOW_COL = 'low'
CLOSE_COL = 'close'
ADJ_CLOSE_COL = 'adj_close'
VOL_COL = 'volume'
TICKER_COL = 'ticker'

REQUIRED_COLS = frozenset({
    DATE_COL,
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    ADJ_CLOSE_COL,
    VOL_COL
})


def rename_bar_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the default return columns from Yahoo to the format that the
    DB expects.

    :param DataFrame df: The ``DataFrame`` that needs the columns renamed.
    :return: The same `DataFrame` passed in but with new column names.
    """
    if set(df.columns) == REQUIRED_COLS:
        return df

    return df.rename(columns={
        'Date': DATE_COL,
        'Open': OPEN_COL,
        'High': HIGH_COL,
        'Low': LOW_COL,
        'Close': CLOSE_COL,
        'Adj Close': ADJ_CLOSE_COL,
        'Volume': VOL_COL
    })


def roll(df: pd.DataFrame, window: int):
    df.dropna(inplace=True)
    roll_array = np.dstack(
            [df.values[i:i + window, :] for i in
             range(len(df.index) - window + 1)]).T
    p = xr.DataArray(roll_array, coords=[df.index[window - 1:],
                                         df.columns,
                                         pd.Index(range(window), name='roll')])
    grouped = p.T.groupby('roll')
    # grouped = p.T.rolling(date=1)
    return grouped

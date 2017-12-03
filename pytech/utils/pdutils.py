from io import StringIO
from typing import Dict

import pandas as pd
import pandas.io.sql
from sqlalchemy.sql.type_api import TypeEngine

from exceptions import (
    DateParsingError,
)

# constants for the expected column names of ALL data DataFrames
DATE_COL = 'date'
OPEN_COL = 'open'
ADJ_OPEN_COL = 'adj_open'
HIGH_COL = 'high'
ADJ_HIGH_COL = 'adj_high'
LOW_COL = 'low'
ADJ_LOW_COL = 'adj_low'
CLOSE_COL = 'close'
ADJ_CLOSE_COL = 'adj_close'
VOL_COL = 'volume'
SPLIT_FACTOR = 'split_factor'
DIVIDEND = 'dividend'
TICKER_COL = 'ticker'
FROM_DB_COL = 'from_db'

COLS = frozenset({
    DATE_COL,
    OPEN_COL,
    ADJ_OPEN_COL,
    HIGH_COL,
    ADJ_HIGH_COL,
    LOW_COL,
    ADJ_LOW_COL,
    CLOSE_COL,
    ADJ_CLOSE_COL,
    VOL_COL,
    SPLIT_FACTOR,
    TICKER_COL,
    DIVIDEND
})


def clean_df(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    Calls :func:`rename_bar_cols` and :func:`parse_date_col`.

    This is just a shortcut to clean a :class:`pd.DataFrame` for use throughout
    the project.
    """
    df = rename_bar_cols(df, ticker=ticker)
    df = parse_date_col(df)
    return df


def parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the :func:`.parse_date` to convert the `DATE_COL` into the standard
    format.

    :param df: The :class:`pd.DataFrame` with a `DATE_COL`.
    :return: The df with a formatted `DATE_COL`.
    :raises: KeyError if there is no `DATE_COL`.
    :raises: DateParsingError if all dates can't be parsed.
    """
    from . import parse_date
    try:
        df[DATE_COL] = df[DATE_COL].apply(parse_date)
    except KeyError as e:
        raise KeyError(f'No {DATE_COL} found.') from e
    except DateParsingError as exc:
        raise DateParsingError('Unable to parse all dates.') from exc
    return df


def rename_bar_cols(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    Rename the default return columns from Yahoo to the format that the
    DB expects.

    :param df: The ``DataFrame`` that needs the columns renamed.
    :param ticker: If provided this will be set as the `TICKER_COL`.
    :return: The same `DataFrame` passed in but with new column names.
    """
    if set(df.columns) == COLS:
        return df

    df = df.rename(columns={
        'Date': DATE_COL,
        'timestamp': DATE_COL,
        'Open': OPEN_COL,
        'High': HIGH_COL,
        'Low': LOW_COL,
        'Close': CLOSE_COL,
        'Adj Close': ADJ_CLOSE_COL,
        'adjusted_close': ADJ_CLOSE_COL,
        'Volume': VOL_COL,
        'dividend amount': DIVIDEND,
        'divCash': DIVIDEND,
        'split coefficient': SPLIT_FACTOR,
        'splitFactor': SPLIT_FACTOR
    })

    if ticker is not None and TICKER_COL not in df.columns:
        df[TICKER_COL] = ticker

    return df


def roll(df: pd.DataFrame, window: int):
    df.dropna(inplace=True)
    for i in range(df.shape[0] - window + 1):
        yield pd.DataFrame(df.values[i:window + i, :],
                           df.index[i:i + window],
                           df.columns)


class PgSQLDataBase(pandas.io.sql.SQLDatabase):
    """A faster implementation of ``panda``'s ``to_sql()``"""

    def to_sql(self,
               frame: pd.DataFrame,
               name: str,
               if_exists: str = 'append',
               index: bool = False,
               index_label: str = None,
               schema: str = None,
               chunksize: int = None,
               dtype: Dict[str, TypeEngine] = None):
        if dtype is not None:
            for col, type_ in dtype.items():
                # noinspection PyTypeChecker
                if not issubclass(type_, TypeEngine):
                    raise TypeError(f'{type_} is not a valid '
                                    f'SQLAlchemy type for col: {col}')
        table = pandas.io.sql.SQLTable(name,
                                       self,
                                       frame=frame,
                                       index=index,
                                       if_exists=if_exists,
                                       index_label=index_label,
                                       schema=self.meta.schema,
                                       dtype=dtype)
        table.create()

        output = StringIO()
        frame.to_csv(output, index=index)
        output.getvalue()
        output.seek(0)
        conn = self.connectable.raw_connection()

        with conn.cursor() as cur:
            cur.copy_from(output, name, sep=',')
            cur.commit()

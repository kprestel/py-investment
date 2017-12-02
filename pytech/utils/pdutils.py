from typing import Dict
from io import StringIO

import pandas as pd
import pandas.io.sql
from pytech.exceptions import PyInvestmentTypeError

# constants for the expected column names of ALL data DataFrames
from sqlalchemy.sql.type_api import TypeEngine

DATE_COL = 'date'
OPEN_COL = 'open'
HIGH_COL = 'high'
LOW_COL = 'low'
CLOSE_COL = 'close'
ADJ_CLOSE_COL = 'adj_close'
VOL_COL = 'volume'
TICKER_COL = 'ticker'
FROM_DB_COL = 'from_db'

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
        'timestamp': DATE_COL,
        'Open': OPEN_COL,
        'High': HIGH_COL,
        'Low': LOW_COL,
        'Close': CLOSE_COL,
        'Adj Close': ADJ_CLOSE_COL,
        'adjusted_close': ADJ_CLOSE_COL,
        'Volume': VOL_COL
    })


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
                    raise PyInvestmentTypeError(f'{type_} is not a valid '
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


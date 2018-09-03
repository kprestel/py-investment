"""
This module holds all database connection functionality.
"""
import logging
import os
from abc import ABCMeta
from io import (
    StringIO,
)
from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    TYPE_CHECKING,
    Union,
    List,
)

import pandas as pd
import psycopg2 as pg
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import (
    Insert,
    insert,
)
from sqlalchemy.exc import (
    IntegrityError,
    OperationalError,
)
from sqlalchemy.sql import (
    Delete,
    Select,
    Update,
)

import pytech.utils as utils
from pytech.data.schema import (
    asset_snapshot,
    bars,
    metadata,
    portfolio,
    portfolio_snapshot,
    assets,
)

if TYPE_CHECKING:
    from fin.portfolio import Portfolio
    from fin.asset.owned_asset import OwnedAsset
    from io import IOBase

dml_stmts = Union[Insert, Update, Delete]

logger = logging.getLogger(__name__)


def getconn():
    """Creates a :class:``psycopg2.pool.ThreadedConnectionPool``"""
    username = os.getenv('PYTECH_USERNAME', 'pytech')
    password = os.getenv('PYTECH_PASSWORD', 'pytech')
    database = os.getenv('PYTECH_DATABASE', 'pytech')
    host = os.getenv('PYTECH_HOST', 'localhost')
    port = os.getenv('PYTECH_PORT', '5432')
    conn = pg.connect(user=username,
                      password=password,
                      database=database,
                      host=host,
                      port=port)
    return conn


# engine = sa.create_engine('postgresql+psycopg2://', creator=getconn)


class sqlaction(metaclass=ABCMeta):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._tables_created = None

    def _create_tables(self) -> None:
        if self._tables_created:
            return
        try:
            metadata.create_all(self.engine, checkfirst=True)
            self._tables_created = True
        except OperationalError:
            self._tables_created = False

    @utils.class_property
    @classmethod
    def engine(cls):
        cls._engine = sa.create_engine('postgresql+psycopg2://', creator=getconn,
                                   echo=True)
        return cls._engine


class write(sqlaction):
    """Performs an ``insert``, ``update``, or ``delete`` statement."""

    def __call__(self,
                 stmt: dml_stmts,
                 vals: Dict[str, Any] = None,
                 *args,
                 **kwargs):
        if not self._tables_created:
            self._create_tables()

        with self.engine.begin() as conn:
            try:
                if vals is None:
                    res = conn.execute(stmt)
                else:
                    res = conn.execute(stmt, vals)
                return res
            except IntegrityError:
                raise

    def df(self, df: pd.DataFrame, table: str, index: bool = False) -> None:
        out_df: pd.DataFrame = df.copy()
        try:
            out_df[utils.VOL_COL] = out_df[utils.VOL_COL].astype(dtype='int64')
        except KeyError:
            self.logger.debug(f"Couldn't find {utils.VOL_COL} in out_df.")

        out_df = out_df.dropna()
        try:
            out_df = out_df[out_df[utils.FROM_DB_COL] == False]
        except KeyError:
            self.logger.debug(f"Couldn't find {utils.FROM_DB_COL} in out_df.")

        out_cols = set(out_df.columns)
        drop_cols = out_cols.difference(utils.COLS)

        out_df: pd.DataFrame = out_df.drop(drop_cols, axis=1)
        output = StringIO()
        out_df.to_csv(output, index=index, header=False)
        output.getvalue()
        output.seek(0)
        ticker = df[utils.TICKER_COL].iat[0]
        ins = (insert(assets).values(ticker=ticker)
            .on_conflict_do_nothing(constraint='asset_pkey'))
        self(ins)

        try:
            self._copy_from(output, table, out_df.columns)
        except pg.IntegrityError as e:
            self._insert_diff(df, e, out_df, table)

    def _insert_diff(self, df: pd.DataFrame,
                     exception: pg.IntegrityError,
                     out_df: pd.DataFrame,
                     table: str):
        """
        Insert the data that is not already in the database.

        :param df: The original :class:`pd.DataFrame`
        :param exception: The original exception. Should be a
            :class:`pg.IntegrityError`.
        :param out_df: The :class:`pd.DataFrame` that was being written to the
            database that caused the original exception.
        :param table: The table to insert the data to.
        """
        if table != 'bar':
            raise NotImplementedError(
                'Fixing IntegrityErrors is only implemented for '
                'the bar table currently.') from exception

        self.logger.info(f'Handling IntegrityError. Attempting to insert new '
                         'rows only.')

        try:
            ticker = df[utils.TICKER_COL].iat[0]
        except KeyError:
            raise KeyError(
                f'Must have {utils.TICKER_COL} to be inserted.') from exception

        q = sa.select([bars]).where(bars.c.ticker == ticker)
        with self.engine.begin() as conn:
            parse_dt_args = {
                'date': {
                    'utc': True,
                    'infer_datetime_format': True
                }
            }
            db_df = pd.read_sql_query(q, conn, parse_dates=parse_dt_args,
                                      index_col=utils.DATE_COL)

        db_df = db_df.dropna(axis=1, how='all')
        db_cols = set(db_df.columns.tolist())

        if db_df.index.name == 'date':
            db_cols.add('date')

        out_cols = set(out_df.columns.tolist())

        if db_cols != out_cols:
            final_df = out_df
        else:
            out_df = out_df.set_index(utils.DATE_COL)
            final_df = out_df[~out_df.index.isin(db_df.index)]

        if final_df.empty:
            self.logger.info('No new data inserted into db.')
            return

        io_ = StringIO()
        final_df = final_df.fillna('NULL')
        if utils.DATE_COL not in final_df.columns:
            if utils.DATE_COL == final_df.index.name:
                final_df.insert(0, utils.DATE_COL, final_df.index)
        final_df.to_csv(io_, index=False, header=False)
        io_.getvalue()
        io_.seek(0)
        try:
            self._copy_from(io_, table, columns=final_df.columns.tolist())
        except pg.IntegrityError as e:
            raise e from exception

    def _copy_from(self,
                   f: StringIO,
                   table: str,
                   columns: List,
                   sep: str = ','):
        # if 'date' is the index it won't be in the columns
        if utils.DATE_COL not in columns:
            columns.insert(0, utils.DATE_COL)
        conn = self.engine.raw_connection()
        try:
            with conn.cursor() as cursor:
                self.logger.debug(f'table: {table}')
                cursor.copy_from(f, table, sep=sep, columns=columns)
                conn.commit()
        except TypeError as e:
            raise e
        finally:
            conn.close()

    def insert_portfolio(self, portfolio_: 'Portfolio',
                         on_conflict: Optional[str] = 'upsert'):
        """
        Insert a portfolio into the DB.

        :param portfolio_: the portfolio to insert.
        :param on_conflict: what to do if a primary key conflict is
            encountered. Valid options are:

            - None
                - do nothing
            - upsert
                - perform an upsert
            - raise
                - raise the :class:`IntegrityError`
        :return:
        """
        ins = insert(portfolio).values(id=portfolio_.id,
                                       cash=portfolio_.cash,
                                       initial_capital=portfolio_.initial_capital)
        if on_conflict == 'upsert':
            ins = ins.on_conflict_do_update(constraint='portfolio_pkey',
                                            set_=dict(cash=portfolio_.cash,
                                                      initial_capital=portfolio_.initial_capital))
        elif on_conflict is None:
            ins = ins.on_conflict_do_nothing(constraint='portfolio_pkey')

        return self(ins)

    def portfolio_snapshot(self, portfolio_: 'Portfolio',
                           cur_dt: 'utils.date_type'):
        """
        Write the portfolio's current state to the `portfolio_snapshot` table.

        :param portfolio_: the portfolio to write
        """
        ins = (insert(portfolio_snapshot).values(portfolio_id=portfolio_.id,
                                                 date=cur_dt,
                                                 cash=portfolio_.cash,
                                                 equity=portfolio_.equity,
                                                 commission=portfolio_.total_commission)
            .on_conflict_do_update(constraint='portfolio_snapshot_pkey',
                                   set_=dict(date=cur_dt,
                                             cash=portfolio_.cash,
                                             equity=portfolio_.equity,
                                             commission=portfolio_.total_commission)))
        return self(ins)

    def owned_asset_snapshot(self, assets: Iterable['OwnedAsset'],
                             portfolio_id: str,
                             cur_dt: 'utils.date_type'):
        with self.engine.begin() as conn:
            conn.execute(
                asset_snapshot.insert(),
                [
                    dict(portfolio_id=portfolio_id,
                         date=cur_dt,
                         ticker=asset.ticker,
                         shares=asset.shares_owned,
                         mv=asset.market_value,
                         close=asset.latest_price)
                    for asset in assets
                ])



class reader(sqlaction):
    def __call__(self, query: Select, ret_df: bool = False, *args, **kwargs):
        with self.engine.begin() as conn:
            if ret_df:
                return pd.read_sql_query(query, conn)
            else:
                for row in conn.execute(query):
                    yield row

    def df(self, query: Select, *args, **kwargs):
        with self.engine.begin() as conn:
            df = pd.read_sql_query(query, conn, index_col='date',
                                   parse_dates={
                                       'date': {
                                           'utc': True,
                                           'format': '%Y-%m-%d %H:%M%S'
                                       }
                                   })
            df = df.sort_index()
            return df

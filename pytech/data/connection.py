"""
This module holds all database connection functionality.
"""
import logging
import os
from abc import ABCMeta
from io import StringIO
from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    TYPE_CHECKING,
    Union,
)

import pandas as pd
import psycopg2 as pg
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import (
    Insert,
    insert,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import (
    Delete,
    Select,
    Update,
)

import pytech.utils as utils
from pytech.data.schema import (
    asset_snapshot,
    portfolio,
    portfolio_snapshot,
    metadata,
    assets,
    bars,
)

if TYPE_CHECKING:
    from fin.portfolio import Portfolio
    from fin.asset.owned_asset import OwnedAsset

dml_stmts = Union[Insert, Update, Delete]


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
    _engine = sa.create_engine('postgresql+psycopg2://', creator=getconn,
                               echo=True)

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        metadata.create_all(self._engine, checkfirst=True)

    @utils.class_property
    @classmethod
    def engine(cls):
        return cls._engine


class write(sqlaction):
    """Performs an ``insert``, ``update``, or ``delete`` statement."""

    def __call__(self,
                 stmt: dml_stmts,
                 vals: Dict[str, Any] = None,
                 *args,
                 **kwargs):
        with self.engine.begin() as conn:
            try:
                if vals is None:
                    res = conn.execute(stmt)
                else:
                    res = conn.execute(stmt, vals)
                return res
            except IntegrityError as e:
                raise

    def df(self, df: pd.DataFrame, table: str, index: bool = False) -> None:
        out_df: pd.DataFrame = df.copy()
        out_df[utils.VOL_COL] = out_df[utils.VOL_COL].astype(dtype='int64')
        out_df = out_df.dropna()
        out_df = out_df[out_df[utils.FROM_DB_COL] == False]
        out_cols = set(out_df.columns)
        req_cols = set()
        req_cols.update(utils.REQUIRED_COLS)
        req_cols.add('ticker')
        drop_cols = out_cols.difference(req_cols)
        drop_cols.add(utils.FROM_DB_COL)
        out_df = out_df.drop(drop_cols, axis=1)
        output = StringIO()
        out_df.to_csv(output, index=index, header=False)
        t = output.getvalue()
        print(t)
        output.seek(0)
        conn = self.engine.raw_connection()
        try:
            with conn.cursor() as cursor:
                cursor.copy_from(output, table, sep=',',
                                 columns=out_df.columns)
                conn.commit()
        except pg.IntegrityError as e:
            try:
                ticker = df[utils.TICKER_COL].iat[0]
                q = sa.select([bars]).where(bars.c.ticker == ticker)
                with self.engine.begin() as conn:
                    db_df = pd.read_sql_query(q, conn)
                tmp_df = pd.concat([db_df, out_df])
                tmp_df = tmp_df.reset_index(drop=True)
                grp_df = tmp_df.groupby(tmp_df.date)
                idx = [x[0] for x in grp_df.groups.values() if len(x) == 1]
                final_df = tmp_df.reindex(idx)
                io_ = StringIO()
                final_df = final_df.fillna('NULL')
                final_df.to_csv(io_, index=index, header=False)
                x = io_.getvalue()
                print(x)
                io_.seek(0)
                conn_ = self.engine.raw_connection()
                with conn_.cursor() as cursor:
                    cursor.copy_from(io_, table, sep=',',
                                     columns=final_df.columns)
                    conn.commit()
            except KeyError:
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

        return self._do_insert(ins)

    def portfolio_snapshot(self, portfolio_: 'Portfolio',
                           cur_dt: 'utils.date_type'):
        """
        Write the portfolio's current state to the `portfolio_snapshot` table.

        :param portfolio_: the portfolio to write
        """
        ins = insert(portfolio_snapshot).values(portfolio_id=portfolio_.id,
                                                date=cur_dt,
                                                cash=portfolio_.cash,
                                                equity=portfolio_.equity,
                                                commission=portfolio_.total_commission)
        return self._do_insert(ins)

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

    def _do_insert(self, ins: Insert):
        with self.engine.begin() as conn:
            res = conn.execute(ins)
            return res


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
            df = pd.read_sql_query(query, conn)
            df.index = df.date
            return df

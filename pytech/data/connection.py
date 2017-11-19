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
    TYPE_CHECKING,
    Union,
    Optional,
)

import pandas as pd
import psycopg2 as pg
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import (
    Delete,
    Select,
    Update,
)
from sqlalchemy.dialects.postgresql import (
    Insert,
    insert,
)

import pytech.utils as utils
from pytech.data.schema import portfolio

if TYPE_CHECKING:
    from fin.portfolio import Portfolio

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

    @utils.class_property
    @classmethod
    def engine(cls):
        return cls._engine


class write(sqlaction):
    """Performs an ``insert``, ``update``, or ``delete`` statement."""

    def __call__(self,
                 stmt: dml_stmts,
                 vals: Dict[str, Any] = None, *args,
                 **kwargs):
        with self.engine.begin() as conn:
            try:
                if vals is None:
                    res = conn.execute(stmt)
                else:
                    res = conn.execute(stmt, vals)
                return res
            except IntegrityError as e:
                self.logger.warning(f'{e}')

    def df(self, df: pd.DataFrame, table: str, index: bool = False) -> None:
        out_df: pd.DataFrame = df.copy()
        out_df[utils.VOL_COL] = out_df[utils.VOL_COL].astype(dtype='int64')
        out_df = out_df.dropna()
        out_df = out_df[out_df[utils.FROM_DB_COL] == False]
        out_df = out_df.drop(utils.FROM_DB_COL, axis=1)
        output = StringIO()
        out_df.to_csv(output, index=index, header=False)
        output.getvalue()
        output.seek(0)
        conn = self.engine.raw_connection()
        try:
            with conn.cursor() as cursor:
                cursor.copy_from(output, table, sep=',',
                                 columns=out_df.columns)
                conn.commit()
        except pg.IntegrityError as e:
            raise
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
        """Writes a portfolio to the database."""
        ins = insert(portfolio).values(id=portfolio_.id,
                                       cash=portfolio_.cash,
                                       initial_capital=portfolio_.initial_capital)
        if on_conflict == 'upsert':
            ins = ins.on_conflict_do_update(constraint='portfolio_pkey',
                                      set_=dict(cash=portfolio_.cash,
                                                initial_capital=portfolio_.initial_capital))
        elif on_conflict is None:
            ins = ins.on_conflict_do_nothing(constraint='portfolio_pkey')

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

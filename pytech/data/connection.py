"""
This module holds all database connection functionality.
"""
from io import StringIO
import pandas as pd
import logging
import os
from abc import ABCMeta
from typing import (
    Any,
    Dict,
    Union,
)

import psycopg2 as pg
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import (
    Insert,
    Delete,
    Update,
    Select,
)

import pytech.utils as utils

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
    _engine = sa.create_engine('postgresql+psycopg2://', creator=getconn)

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

    def df(self, df: pd.DataFrame, table: str, index: bool = False):
        output = StringIO()
        df[utils.VOL_COL] =  df[utils.VOL_COL].astype(dtype='int64')
        df.to_csv(output, index=index, header=False)
        output.getvalue()
        output.seek(0)
        conn = self.engine.raw_connection()
        try:
            with conn.cursor() as cursor:
                cursor.copy_from(output, table, sep=',', columns=df.columns)
                conn.commit()
        finally:
            conn.close()


class reader(sqlaction):
    def __call__(self, query: Select, ret_df: bool = False, *args, **kwargs):
        print('penis')
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


import logging
from abc import ABCMeta, abstractmethod

import pandas as pd
from sqlalchemy import MetaData, create_engine, not_, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import InvalidRequestError

from pytech.db.connector import DBConnector
from pytech.db.pytech_db_schema import (PYTECH_DB_TABLE_NAMES,
                                        asset as asset_table,
                                        universe_ohlcv as ohlcv_table)
from pytech.fin.asset.asset import Asset
from pytech.utils.exceptions import AssetNotInUniverseError


class Finder(metaclass=ABCMeta):
    """
    Abstract Base Class that provides an interface for any form of database access.

    Each subclass is responsible for being able to find rows in the database that correspond to one class or
    class family like the :class:`Asset`.

    .. note:
        * This class is **NOT** meant to ever be instantiated
        * Subclasses must implement the following methods:
            * find_instance
        * Subclasses **MUST** call the :class:`Finder` constructor
    """

    def __init__(self, engine=None, **kwargs):
        """
        **ALL** subclasses must call this constructor in the first line of
        their constructor.

        :param engine: An engine with a connection to the database that
            contains the data to find.
            This parameter can either be a SQLAlchemy.engine or a string URI
            that can be turned into a SQLAlchemy engine.
            If a string is passed in ``**kwargs`` will be passed into the
            :func:`sqlalchemy.create_engine` method.
            Check out the SQLAlchemy docs for more information about
            possible extra parameters.
        :type engine: :class:`SQLAlchemy.engine` or str
        """
        # self.connector = DBConnector(engine, **kwargs)
        # self.engine = self.connector.engine
        # self.conn = self.engine.connect()

        # metadata = MetaData(bind=self.engine)

        # metadata.reflect(only=PYTECH_DB_TABLE_NAMES)

        # for table_name in PYTECH_DB_TABLE_NAMES:
        #     setattr(self, table_name, metadata.tables[table_name])

        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def find_instance(self, key):
        """
        This method should should return **ONE** instance of the class
        that the :class:`Finder` instance is responsible for finding.

        :param key: This should be the unique identifier for the specific
        instance that is needed.
        :return: The instance of the class requested.
        """
        raise NotImplementedError('find_instance')

    @abstractmethod
    def find_all(self):
        """
        This method should return an iterable containing all instances of the
        class that the ':class:`Finder` instance is responsible for finding.

        If applicable or possible the return object should be a dictionary
        where the key is the ticker corresponding to whatever object is being
        found and the value should be that object.

        :return: dict[ticker -> Asset]
        :rtype: dict
        """
        raise NotImplementedError('find_all')

    @staticmethod
    def row_to_dict(row):
        """
        Convert one row from a ``SQLAlchemy`` query to a dictionary.

        :param sqlalchmey.RowProxy row: A ``RowProxy`` object that is returned
            from a ``SQLAlchemy`` query.
        :return: The dictionary that is the row.
        :rtype: dict
        """

        # may need if k doesn't start with '_'
        return {k: v for k, v in row.items()}


class AssetFinder(Finder):
    """
    Provides an interface to the DB to find assets based on the :class:`pytech.asset.Asset`'s **ticker**.
    """

    def __init__(self, engine=None, **kwargs):
        super().__init__(engine, **kwargs)
        self.asset_class_dict = Asset.get_subclass_dict()
        self._asset_cache = {}
        self._ohlcv_cache = {}

    def find_instance(self, key):
        """
        Find and return an instance of an :class:`Asset` including
        the OHLCV ``DataFrame``.

        The ``asset_class_dict`` will be used to call the correct constructor
        based on the ``asset_type`` found.

        :param str key: The **ticker** of the asset to retrieve.
        :return: The ``Asset`` that corresponds to the ``key``
        :rtype: Asset
        :raises: AssetNotInUniverseError when an asset cannot be found in the
            database with the requested ``key``.
        """
        asset = self._asset_cache.get(key)

        if asset is not None:
            self.logger.debug(
                    'Found asset with ticker: {} in cache.'.format(key))
            return asset

        sql = select([self.asset]).where(asset_table.c.ticker == key)

        result = self.conn.execute(sql)
        row = result.first()

        if row is None:
            raise AssetNotInUniverseError(ticker=key)

        asset_dict = self.row_to_dict(row)
        asset_dict['data'] = self.find_ohlcv(ticker=key)

        asset_type_class = self.asset_class_dict.get(asset_dict['asset_type'],
                                                     Asset)

        asset = asset_type_class.from_dict(asset_dict)

        # update caches
        self._asset_cache[asset.ticker] = asset_dict['data']
        self._asset_cache[asset.ticker] = asset

        return asset

    def find_all(self):
        """
        Retrieve all :class:`pytech.asset.Asset` in the database as a dictionary.

        :return: A dictionary where the ``key`` = the ticker and the
            ``value`` = the :class:`pytech.asset.Asset` instance
        """
        sql = select([self.asset]).where(
                not_(asset_table.c.ticker.in_(self._asset_cache.keys())))

        all_assets_dict = self._asset_cache

        for row in self.conn.execute(sql).fetchall():
            asset_dict = self.row_to_dict(row)
            asset_dict['data'] = self.find_ohlcv(asset_dict['ticker'])
            asset_type_class = self.asset_class_dict.get(
                    asset_dict['asset_type'], Asset)
            temp_asset = asset_type_class.from_dict(asset_dict)
            all_assets_dict[temp_asset.ticker] = temp_asset

        return all_assets_dict

    def find_ohlcv(self, ticker):
        """
        Find and return the OHLCV `pandas.DataFrame` for the requested ticker.

        :param str ticker: The ticker of the asset to retrieve the OHLCV for.
        :return: The OHLCV DataFrame
        :rtype: :class:`pandas.DataFrame`
        """
        df = self._ohlcv_cache.get(ticker)

        if df is not None:
            return df

        sql = select([self.universe_ohlcv]).where(
                ohlcv_table.c.ticker == ticker)
        df = pd.read_sql_query(sql, self.engine, index_col='asof_date',
                               parse_dates={
                                   'asof_date': 'ns'
                               })

        self._ohlcv_cache[ticker] = df

        return df

from abc import ABCMeta, abstractmethod

from sqlalchemy import MetaData
from sqlalchemy import create_engine
from pytech.db.pytech_db_schema import metadata, PYTECH_DB_TABLE_NAMES


class Finder(metaclass=ABCMeta):
    """
    Abstract Base Class that provides an interface for any form of database access.

    Each subclass is responsible for being able to find rows in the database that correspond to one class or
    class family like the :class:`Asset`.

    .. note:
        * This class is **NOT** meant to ever be instantiated
        * Subclasses must implement the following methods:
            * find
        * Subclasses **MUST** call the :class:`Finder` constructor
    """

    def __init__(self, engine, **kwargs):
        """
        **ALL** subclasses must call this constructor in the first line of their constructor.

        :param engine: An engine with a connection to the database that contains the data to find.
            This parameter can either be a SQLAlchemy.engine or a string URI that can be turned into a SQLAlchemy engine.
            If a string is passed in ``**kwargs`` will be passed into the :func:`sqlalchemy.create_engine` method.
            Check out the SQLAlchemy docs for more information about possible extra parameters.
        :type engine: :class:`SQLAlchemy.engine` or str
        """

        if isinstance(engine, str):
            self.engine = create_engine(str, **kwargs)
        else:
            self.engine = engine

        metadata = MetaData(bind=self.engine)
        metadata.reflect(only=PYTECH_DB_TABLE_NAMES)

        for table_name in PYTECH_DB_TABLE_NAMES:
            setattr(self, table_name, metadata.tables[table_name])

    @abstractmethod
    def find_instance(self, key):
        """
        This method should should return **ONE** instance of the class that the :class:`Finder` instance is responsible
        for finding.

        :param key: This should be the unique identifier for the specific instance that is needed.
        :return: The instance of the class requested.
        """
        raise NotImplementedError('find_instance')


class AssetFinder(object):
    """
    Provides an interface to the DB to find assets based on either **id** or **ticker**.
    """
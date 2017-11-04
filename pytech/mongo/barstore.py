import logging
from typing import Any, Dict, Union

import pandas as pd
from arctic.chunkstore._chunker import Chunker
from arctic.chunkstore.chunkstore import ChunkStore
from arctic.chunkstore.date_chunker import DateChunker
from arctic.date import DateRange
from arctic.decorators import mongo_retry

import pytech.utils as utils


class BarStore(ChunkStore):
    """Override the required methods so that they can be wrapped properly."""

    LIBRARY_TYPE = 'BAR_STORE'
    LIBRARY_NAME = 'pytech.bars'

    def __init__(self, arctic_lib):
        self.logger = logging.getLogger(__name__)
        super().__init__(arctic_lib)
        self.logger.info(f'BarStore collection name: {arctic_lib.get_name()}')

    # @mongo_retry
    def read(self, symbol: str,
             chunk_range: pd.DatetimeIndex or DateRange = None,
             filter_data=True,
             **kwargs) -> pd.DataFrame or pd.Series:
        """
        Retrieve data from the DB.

        :rtype:
        :param symbol: The key for the data you wish to retrieve.
        :param chunk_range: This depends on the :class:``Chunker`` used.
        If using a :class:``DateChunker`` then a range of dates can be given in
        order to only retrieve data in that range.
        :param filter_data:
        :param kwargs:
        :return:
        """
        cols = kwargs.pop('columns', None)
        if cols is not None and not isinstance(cols, list):
            cols = list(cols)

        return super().read(symbol, chunk_range, filter_data, columns=cols,
                            **kwargs)

    @mongo_retry
    def write(self, symbol: str,
              item: Union[pd.DataFrame, pd.Series],
              metadata: Any = None,
              chunker: Chunker = DateChunker(),
              audit: Dict = None,
              **kwargs) -> None:
        """
        Write and replace data in the DB.

        :param symbol: The symbol that will be used to store/access data in the DB.
        :param item: The :class:``pd.DataFrame`` or :class:``pd.Series`` that will
        be stored in the DB.
        :param metadata: optional per symbol metadata
        :param chunker: The Arctic chunker that should be used to chunk the
        data.
        :param audit: Audit information.
        :param kwargs: All of these will be passed onto the ``chunker``.
        In the case of the default :class:``DateChunker`` you can specify a
        ``chunk_size`` (D, M, or Y).
        """
        if not isinstance(item, (pd.DataFrame, pd.Series)):
            raise TypeError('Can only chunk DataFrames and Series. '
                            f'{type(item)} was provided')

        # ensure that the column names are correct before writing it.
        item = utils.rename_bar_cols(item)

        return super().write(symbol, item, metadata, chunker, audit, **kwargs)

    @mongo_retry
    def delete(self, symbol: str,
               chunk_range: pd.DatetimeIndex or DateRange = None,
               audit: Dict = None) -> None:
        """
        Delete all chunks for a symbol, or optionally, chunks within a range.

        :param symbol: The key of the item.
        :param chunk_range: A date range to delete.
        :param audit: A dict to store in the audit log.
        """
        return super().delete(symbol, chunk_range, audit)

    # @mongo_retry
    def update(self, symbol: str,
               item: Union[pd.DataFrame, pd.Series],
               metadata: Any = None,
               chunk_range: Union[pd.DatetimeIndex, DateRange] = None,
               upsert: bool = False,
               audit: Dict[Any, Any] = None,
               **kwargs) -> None:
        """
        Overwrites data in the DB with data in item for the given item.

        Is idempotent.

        :param symbol: The key in the db.
        :param item: The data to update.
        :param chunk_range: If a range is specified, it will clear/delete
        the data within the range and overwrite it with the data in item.
        This allows the user to update with data that might only be a subset of the
        original data.
        :param upsert: If True then will write the data even if the symbol does
        not exist.
        :param metadata: optional per symbol metadata
        :param audit: Audit information.
        :param kwargs: optional keywords passed to write during an upsert.

            * chunk_size
            * chunker

        """
        return super().update(symbol, item, metadata, chunk_range, upsert,
                              audit, **kwargs)

    @mongo_retry
    def append(self, symbol: str,
               item: pd.DataFrame or pd.Series,
               metadata: Any = None,
               audit: Dict = None) -> None:
        """
        Appends data from ``item`` to symbol's data in the DB.

        Is not idempotent.

        :param symbol: The key in the DB.
        :param item: The data to append.
        :param metadata: optional symbol metadata.
        :param audit: Audit information.
        """
        return super().append(symbol, item, metadata, audit)

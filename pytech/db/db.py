"""
This module is meant to be a wrapper around the Arctic library which can be
 found here https://github.com/manahl/arctic/
 
 It is contained in a separate module in order to initialize the db properly.
 """
from typing import Dict

import pandas as pd
from arctic import Arctic, CHUNK_STORE
from arctic.chunkstore.chunkstore import ChunkStore
from arctic.chunkstore.date_chunker import DateChunker
from arctic.date import DateRange
from arctic.decorators import mongo_retry

a = Arctic('localhost')
a.initialize_library('pytech', lib_type=CHUNK_STORE)
lib = a['pytech']  # type: ChunkStore


@mongo_retry
def write(symbol: str,
          item: pd.DataFrame or pd.Series,
          chunker=DateChunker(),
          **kwargs) -> None:
    """
    Write and replace data in the DB.
    
    :param symbol: The symbol that will be used to store/access data in the DB.
    :param item: The :class:``pd.DataFrame`` or :class:``pd.Series`` that will
    be stored in the DB.
    :param chunker: The Arctic chunker that should be used to chunk the data.
    :param kwargs: All of these will be passed onto the ``chunker``.  
    In the case of the default :class:``DateChunker`` you can specify a 
    ``chunk_size`` (D, M, or Y).
    """
    lib.write(symbol, item, chunker=chunker, **kwargs)


@mongo_retry
def read(symbol: str,
         chunk_range: pd.DatetimeIndex or DateRange = None,
         filter_data=True,
         **kwargs) -> pd.DataFrame or pd.Series:
    """
    Retrieve data from the DB.
    
    :param symbol: The key for the data you wish to retrieve. 
    :param chunk_range: This depends on the :class:``Chunker`` used. 
    If using a :class:``DateChunker`` then a range of dates can be given in 
    order to only retrieve data in that range.
    :param filter_data: 
    :param kwargs: 
    :return: 
    """
    return lib.read(symbol,
                    chunk_range=chunk_range,
                    filter_data=filter_data,
                    **kwargs)


@mongo_retry
def append(symbol: str, item: pd.DataFrame or pd.Series) -> None:
    """
    Appends data from ``item`` to symbol's data in the DB.
    
    Is not idempotent.
    
    :param symbol: The key in the DB.
    :param item: The data to append.
    """
    lib.append(symbol, item)


@mongo_retry
def update(symbol: str, item: pd.Series or pd.DataFrame,
           chunk_range: pd.DatetimeIndex or DateRange = None,
           upsert: bool = False,
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
    :param kwargs: optional keywords passed to write during an upsert.
    
        * chunk_size
        * chunker
        
    """
    lib.update(symbol, item, chunk_range=chunk_range, upsert=upsert, **kwargs)


@mongo_retry
def delete(symbol: str,
           chunk_range: pd.DatetimeIndex or DateRange = None,
           audit: Dict = None) -> None:
    """
    Delete all chunks for a symbol, or optionally, chunks within a range.
    
    :param symbol: The key of the item.
    :param chunk_range: A date range to delete.
    :param audit: A dict to store in the audit log.
    """
    lib.delete(symbol, chunk_range, audit)

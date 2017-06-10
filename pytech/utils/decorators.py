from functools import wraps
import pandas as pd
from arctic import arctic

from arctic.chunkstore.chunkstore import ChunkStore

import pytech.utils.pandas_utils as pd_utils
from pytech.mongo import ARCTIC_STORE, BarStore
from pytech.utils.exceptions import InvalidStoreError, PyInvestmentKeyError
from pandas.tseries.offsets import BDay


def memoize(obj):
    """Memoize functions so they don't have to be reevaluated."""
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer


def write_chunks(lib_name, chunk_size='D', remove_ticker=True):
    """
    Used to wrap functions that return :class:`pd.DataFrame`s and writes the
    output to a :class:`ChunkStore`. It is required that the the wrapped
    function contains a column called 'ticker' to use as the key in the db.

    :param lib_name: The name of the library to write the
        :class:`pd.DataFrame` to.
    :param chunk_size: The chunk size to use options are:

        * D = Days
        * M = Months
        * Y = Years

    :param remove_ticker: If true the ticker column will be deleted before the
        :class:`pd.DataFrame` is returned, otherwise it will remain.
    :return: The output of the original function.
    """
    def wrapper(f):
        @wraps(f)
        def eval_and_write(*args, **kwargs):
            df = f(*args, **kwargs)
            try:
                # TODO: make this use the fast scalar getter
                ticker = df[pd_utils.TICKER_COL][0]
                # ticker = df.at[0, pd_utils.TICKER_COL]
            except KeyError:
                raise PyInvestmentKeyError(
                        'Decorated functions are required to add a column '
                        f'{pd_utils.TICKER_COL} that contains the ticker.')

            if remove_ticker:
                # should this be saved?
                df.drop(pd_utils.TICKER_COL, axis=1, inplace=True)

            if lib_name not in ARCTIC_STORE.list_libraries():
                # create the lib if it does not already exist
                ARCTIC_STORE.initialize_library(lib_name,
                                                BarStore.LIBRARY_TYPE)

            lib = ARCTIC_STORE[lib_name]

            if not isinstance(lib, ChunkStore):
                raise InvalidStoreError(required=ChunkStore,
                                        provided=type(lib))
            else:
                lib.update(ticker, df, chunk_size=chunk_size, upsert=True)

            df.index.freq = BDay()
            return df
        return eval_and_write
    return wrapper


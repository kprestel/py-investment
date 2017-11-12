from functools import wraps

import pandas as pd
from arctic.chunkstore.chunkstore import ChunkStore
from pandas.tseries.offsets import BDay
from sqlalchemy import Table

import pytech.utils as utils
from pytech.data.schema import assets
from pytech.data.connection import write
from pytech.exceptions import (
    InvalidStoreError,
    PyInvestmentKeyError,
)
from pytech.data._holders import ReaderResult
from pytech.mongo import (
    ARCTIC_STORE,
    BarStore,
)


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


def optional_arg_decorator(fn):
    """Used to **only** to wrap decorators that take optional arguments."""

    def wrapped_decorator(*args):
        if len(args) == 1 and callable(args[0]):
            return fn(args[0])

        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args)

            return real_decorator

    return wrapped_decorator


def write_df(table: str):
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
        :class:`pd.DataFrame` is returned, otherwise it will remain which is
        going to use more memory than required.
    :return: The output of the original function.
    """

    def wrapper(f):
        @wraps(f)
        def eval_and_write(*args, **kwargs):
            result = f(*args, **kwargs)
            df = result.df
            try:
                # TODO: make this use the fast scalar getter
                # ticker = df[utils.TICKER_COL][0]
                ticker = df[utils.TICKER_COL].iat[0]
                # ticker = df.at[0, pd_utils.TICKER_COL]
            except KeyError:
                raise PyInvestmentKeyError(
                    'Decorated functions are required to add a column '
                    f'"{utils.TICKER_COL}" that contains the ticker.')

            if 'date' not in df.columns and 'date' in df.index.names:
                df[utils.DATE_COL] = df.index

            # this is a work around for a flaw in the the arctic DateChunker.
            # if 'date' not in df.columns or 'date' not in df.index.names:
            #     if df.index.dtype == pd.to_datetime(['2017']).dtype:
            #         df.index.name = 'date'
            #     else:
            #         raise ValueError('df must be datetime indexed or have a'
            #                          'column named "date".')

            # if lib_name not in ARCTIC_STORE.list_libraries():
                # create the lib if it does not already exist
                # ARCTIC_STORE.initialize_library(lib_name,
                #                                 BarStore.LIBRARY_TYPE)

            # lib = ARCTIC_STORE[lib_name]

            # if not isinstance(lib, ChunkStore):
            #     raise InvalidStoreError(required=ChunkStore,
            #                             provided=type(lib))
            # else:
            #     lib.update(ticker, df, chunk_size=chunk_size, upsert=True)

            writer = write()

            ins = assets.insert().values(ticker=ticker)
            writer(ins)

            df.index.freq = BDay()
            writer.df(df, table)
            return ReaderResult(ticker, df)

        return eval_and_write

    return wrapper


class lazy_property(object):
    """
    Used for lazy evaluation of an obj attr.

    Property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, f):
        self.f = f
        self.func_name = f.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        val = self.f(obj)
        setattr(obj, self.func_name, val)
        return val

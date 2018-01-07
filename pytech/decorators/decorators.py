import logging
from functools import wraps

from pandas.tseries.offsets import BDay
from psycopg2._psycopg import IntegrityError
from sqlalchemy.dialects.postgresql import insert

import pytech.utils as utils
from pytech.exceptions import (
    PyInvestmentKeyError,
)

logger = logging.getLogger(__name__)


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
    Used to wrap functions that return a :class:`pd.DataFrame` that want the
    result to be persisted to the db.

    :param table: the name of the table to write the df to.
    :return: the result of the function
    """
    from pytech.data._holders import ReaderResult
    from pytech.data import write
    from pytech.data.schema import assets

    def wrapper(f):
        @wraps(f)
        def eval_and_write(*args, **kwargs):
            result = f(*args, **kwargs)

            try:
                if not result.successful:
                    return result
                df = result.df
            except AttributeError:
                df = result

            try:
                # TODO: make this use the fast scalar getter
                ticker = df[utils.TICKER_COL].iat[0]
            except KeyError:
                raise PyInvestmentKeyError(
                    'Decorated functions are required to add a column '
                    f'"{utils.TICKER_COL}" that contains the ticker.')

            if 'date' not in df.columns and 'date' in df.index.names:
                df[utils.DATE_COL] = df.index
            writer = write()
            ins = (insert(assets).values(ticker=ticker)
                .on_conflict_do_nothing(constraint='asset_pkey'))
            writer(ins)

            df.index.freq = BDay()
            try:
                writer.df(df, table)
            except IntegrityError as e:
                logger.warning(f'Unable to insert df. {e.pgerror}')

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

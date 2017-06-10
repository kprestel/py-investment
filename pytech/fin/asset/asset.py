import datetime as dt
import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

import pytech.data.reader as reader
import pytech.utils.dt_utils as dt_utils
import pytech.utils.pandas_utils as pd_utils
from pytech.fin.market_data.market import Market
from pytech.utils.decorators import memoize


def _calc_beta(df: pd.DataFrame) -> pd.Series:
    """
    Calculates beta given a :class:`pd.DataFrame`.
    It is expected that the df has the stock returns are in column 0 and the
    market returns in column 1.
    """
    x = df.values[:, [1]]
    # noinspection PyUnresolvedReferences
    x = np.concatenate([np.ones_like(x), x], axis=1)
    beta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(df.values[:, 0:])
    return pd.Series(beta[1][0])


class Asset(metaclass=ABCMeta):
    """
    This is the base class that all Asset classes should inherit from.

    Inheriting from it will provide a table name and the proper mapper args
    required for the db.  It will also allow it to have a relationship
    with the :class:``OwnedAsset``.

    The child class is responsible for giving each instance a ticker to identify it.

    If the child class needs any more fields it is responsible for creating
    them at the class level as well as populating them via the child's constructor,
    in addition to calling the ``Asset`` constructor.

    Any child class instance of this base class is considered to be a part of
    the **Asset Universe** or the assets that
    are eligible to be traded.  If a child instance of an Asset does not yet
    exist in the universe and the
    :class:``~pytech.portfolio.Portfolio`` tries to trade it an exception will occur.
    """

    def __init__(self, ticker: str, start_date: dt.datetime,
                 end_date: dt.datetime):
        self.ticker = ticker
        self.asset_type = self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__name__)

        start_date, end_date = dt_utils.sanitize_dates(start_date, end_date)

        self.start_date = start_date
        self.end_date = end_date
        self.market = Market(start_date=self.start_date,
                             end_date=self.end_date)

        if self.start_date >= self.end_date:
            raise ValueError(
                    'start_date must be older than end_date. '
                    'start_date: {} end_date: {}'.format(
                            str(start_date),
                            str(end_date)))

    @property
    def data(self):
        return self.get_data()

    @data.setter
    def data(self, ohlcv):
        if not (isinstance(ohlcv, pd.DataFrame) or
                    not isinstance(ohlcv, pd.Series)):
            raise TypeError(
                    'data must be a pandas DataFrame or TimeSeries. '
                    '{} was provided'.format(type(ohlcv)))

        self._ohlcv = ohlcv

    @classmethod
    def get_subclass_dict(cls, subclass_dict=None):
        """
        Get a dictionary of subclasses for :class:`Asset` where the key is the string name of the class and the value
        is the actual class reference.

        :param dict subclass_dict: This is used for recursion to maintain the subclass_dict through each call.
        :return: A dictionary where the key is the string name of the subclass and the value is the reference to the class
        :rtype: dict
        """

        if subclass_dict is None:
            subclass_dict = {}
        else:
            subclass_dict = subclass_dict

        for subclass in cls.__subclasses__():
            # prevent duplicate keys
            if subclass.__name__ not in subclass_dict:
                subclass_dict[subclass.__name__] = subclass
                subclass.get_subclass_dict(subclass_dict)
        return subclass_dict

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Must return a :class:`pd.DataFrame` with ticker data."""
        raise NotImplementedError


class Stock(Asset):
    def __init__(self, ticker: str, start_date: dt.datetime,
                 end_date: dt.datetime, source: str = 'google'):
        self.source = source
        super().__init__(ticker, start_date, end_date)

    @memoize
    def get_data(self) -> pd.DataFrame:
        d = reader.get_data(self.ticker, self.source,
                            self.start_date, self.end_date)
        return d[self.ticker]

    # noinspection PyTypeChecker
    def calculate_beta(self, col=pd_utils.CLOSE_COL) -> pd.DataFrame:
        stock_pct_change = self.data[col].pct_change()
        mkt_pct_change = self.market.data[col].pct_change()
        df = pd.concat([stock_pct_change, mkt_pct_change], axis=1)
        rolling = pd_utils.roll(df, 12)
        betas = pd.concat([_calc_beta(sdf) for sdf in pd_utils.roll(df, 12)],
                          axis=1).T
        return betas

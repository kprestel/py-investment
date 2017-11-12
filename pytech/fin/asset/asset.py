import datetime as dt
import logging
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd

import pytech.utils as utils
from pytech.data._holders import ReaderResult
from pytech.decorators.decorators import memoize, write_chunks
from pytech.data.reader import BarReader
from pytech.fin.market.market import Market

BETA_STORE = 'pytech.beta'


def _calc_beta(df: pd.DataFrame) -> pd.Series:
    """
    Calculates beta given a :class:`pd.DataFrame`.
    It is expected that the df has the stock returns are in column 1 and the
    market returns in column 0.
    """
    x = df.values[:, [0]]
    # noinspection PyUnresolvedReferences
    x = np.concatenate([np.ones_like(x), x], axis=1)
    beta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(df.values[:, 1:])
    return pd.Series(beta[1], df.columns[1:], name=df.index[-1])


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

    def __init__(self,
                 ticker: str,
                 start_date: dt.datetime,
                 end_date: dt.datetime):
        self.ticker = ticker
        self.asset_type = self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__name__)

        start_date, end_date = utils.sanitize_dates(start_date, end_date)

        self.start_date = start_date
        self.end_date = end_date
        self.market = Market(start_date=self.start_date,
                             end_date=self.end_date)

        if self.start_date >= self.end_date:
            raise ValueError('start_date must be older than end_date. '
                             f'start_date: {start_date} end_date: {end_date}.')

    @property
    def df(self):
        return self.get_data()

    @df.setter
    def df(self, ohlcv):
        if isinstance(ohlcv, pd.DataFrame) or isinstance(ohlcv, pd.Series):
            self._ohlcv = ohlcv
        else:
            raise TypeError('data must be a pandas DataFrame or Series. '
                            f'{type(ohlcv)} was provided.')

    @classmethod
    def get_subclass_dict(cls, subclass_dict=None):
        """
        Get a dictionary of subclasses for :class:`Asset` where the key is
        the string name of the class and the value is the actual class
        reference.

        :param dict subclass_dict: This is used for recursion to maintain the
            subclass_dict through each call.
        :return: A dictionary where the key is the string name of the
            subclass and the value is the reference to the class
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
                 end_date: dt.datetime, source: str = 'google',
                 lib_name: str = 'pytech.bars'):
        self.source = source
        self.reader = BarReader(lib_name)
        self.lib_name = lib_name
        super().__init__(ticker, start_date, end_date)

    @memoize
    def get_data(self) -> pd.DataFrame:
        return self.reader.get_data(self.ticker, self.source,
                                    self.start_date, self.end_date)

    def last_price(self, col=utils.CLOSE_COL):
        return self.df[col][-1]

    @write_chunks()
    def _rolling_beta(self,
                     col=utils.CLOSE_COL,
                     window: int = 30) -> ReaderResult:
        """
        Calculate the rolling beta over a given window.

        :param col: The column to use to get the returns.
        :param window: The window to use to calculate the rolling beta (days)
        :return: A DataFrame with the betas.
        """
        stock_pct_change = pd.DataFrame(self.returns(col))
        mkt_pct_change = pd.DataFrame(self.market.market[col].pct_change())
        df: pd.DataFrame = pd.concat([mkt_pct_change, stock_pct_change],
                                     axis=1)
        betas = pd.concat([_calc_beta(sdf)
                           for sdf in utils.roll(df, window)], axis=1).T
        betas['ticker'] = self.ticker
        return ReaderResult(self.ticker, betas)

    def rolling_beta(self,
                      col=utils.CLOSE_COL,
                      window: int = 30) -> pd.DataFrame:
        """
        Calculate the rolling beta over a given window.

        This is a wrapper around `_rolling_beta` to return just a dataframe.

        :param col: The column to use to get the returns.
        :param window: The window to use to calculate the rolling beta (days)
        :return: A DataFrame with the betas.
        """
        df_lib_name = self._rolling_beta(col, window)
        return df_lib_name.df

    def returns(self, col=utils.CLOSE_COL) -> pd.Series:
        return self.df[col].pct_change()

    def avg_return(self, col=utils.CLOSE_COL):
        ret = self.returns(col).mean()
        return ret * 252

    def cagr(self, col=utils.CLOSE_COL):
        """Compounding annual growth rate."""
        days = (self.df.index[-1] - self.df.index[0]).days
        return ((self.df[col][-1] / self.df[col][1]) ** (365.0 / days)) - 1

    def std(self, col=utils.CLOSE_COL):
        """Standard deviation of returns, *annualized*."""
        return self.returns(col).std() * np.sqrt(252)

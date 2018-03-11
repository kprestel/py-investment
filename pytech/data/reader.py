"""
Act as a wrapper around pandas_datareader and write the responses to the
database to be accessed later.
"""
import datetime as dt
import itertools
import logging
from multiprocessing.pool import ThreadPool as Pool
from typing import (
    Dict,
    Iterable,
    List,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import sqlalchemy as sa
from pandas.tseries.offsets import BDay

import pytech.utils as utils
from pytech.data._holders import ReaderResult
from pytech.data.connection import reader
from pytech.data.schema import (
    assets,
    bars,
)
from pytech.decorators import write_df
from pytech.exceptions import DataAccessError
from pytech.sources import (
    AlphaVantageClient,
    TiingoClient,
)
from ..utils import DateRange
from pytech.sources.restclient import RestClientError

logger = logging.getLogger(__name__)

ticker_input = TypeVar('A', Iterable, str, pd.DataFrame)
range_type = TypeVar('A', pd.DatetimeIndex, DateRange)

YAHOO = 'yahoo'
GOOGLE = 'google'
FRED = 'fred'
FAMA_FRENCH = 'famafrench'
TIINGO = 'tiingo'
ALPHA_VANTAGE = 'alpha_vantage'
_CLIENTS = {TIINGO, ALPHA_VANTAGE}


class BarReader(object):
    """Read and write data from the DB and the web."""

    def __init__(self):
        self.reader = reader()
        self.tiingo = TiingoClient()
        self.alpha_vantage = AlphaVantageClient()

    @property
    def tickers(self):
        q = sa.select([assets.c.ticker])
        for s in self.reader(q):
            yield s

    def get_data(self,
                 tickers: ticker_input,
                 source: str = YAHOO,
                 date_range: DateRange = None,
                 check_db: bool = True,
                 **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get data and create a :class:`pd.DataFrame` from it.

        :param tickers: The ticker(s) that data will be retrieved for.
        :param source: The data source.
        :param date_range: The :class:`DateRange` to get data for.
        :param check_db: Check the database first before making network call.
        :param filter_data: Filter data from the DB. Only used if `check_db` is
            `True`.
        :param kwargs: kwargs are passed blindly to `pandas_datareader`
        :return: a :class:`pd.DataFrame` if only one ticker was requested or a dictionary
            of :class:`pd.DataFrame`s where the key is the ticker associated with the
            :class:`pd.DataFrame`
        """
        date_range = date_range or DateRange()

        if isinstance(tickers, str):
            try:
                result = self._single_get_data(tickers, source, date_range,
                                               check_db, **kwargs)
                return result.df
            except DataAccessError:
                raise RestClientError(f'Could not get data for ticker: '
                                      f'{tickers}')
        else:
            if isinstance(tickers, pd.DataFrame):
                tickers = tickers.index
            try:
                return self._mult_tickers_get_data(tickers, source, date_range,
                                                   check_db, **kwargs)
            except DataAccessError:
                raise

    def _mult_tickers_get_data(self,
                               tickers: List,
                               source: str,
                               date_range: DateRange,
                               check_db: bool,
                               **kwargs) -> Dict[str, pd.DataFrame]:
        """Download data for multiple tickers."""
        stocks = {}
        failed = []
        passed = []

        with Pool(len(tickers)) as pool:
            result = pool.starmap_async(self._single_get_data,
                                        zip(tickers,
                                            itertools.repeat(source),
                                            itertools.repeat(date_range),
                                            itertools.repeat(check_db)))
            res = result.get()

        for r in res:
            if r.successful:
                stocks[r.ticker] = r.df
                passed.append(r.ticker)
            else:
                failed.append(r.ticker)

        if len(passed) == 0:
            raise DataAccessError('No data could be retrieved.')

        if len(stocks) > 0 and len(failed) > 0 and len(passed) > 0:
            df_na = stocks[passed[0]].copy()
            df_na[:] = np.nan
            for t in failed:
                logger.warning(f'No data could be retrieved for ticker: {t}, '
                               f'replacing with NaN.')
                stocks[t] = df_na

        return stocks

    def _single_get_data(self,
                         ticker: str,
                         source: str,
                         date_range: DateRange,
                         check_db: bool,
                         **kwargs) -> ReaderResult:
        """Do the get data method for a single ticker."""
        if check_db:
            try:
                return self._from_db(ticker, source, date_range, **kwargs)
            except DataAccessError:
                # don't raise, try to make the network call
                logger.info(f'Ticker: {ticker} not found in DB, attempting to'
                            f'download data for {ticker}')

        try:
            return self._from_web(ticker, source, date_range, **kwargs)
        except RestClientError:
            logger.warning(f'Error getting data from {source} '
                           f'for ticker: {ticker}', exc_info=1)
            return ReaderResult(ticker, successful=False)

    @write_df('bar')
    def _from_web(self,
                  ticker: str,
                  source: str,
                  date_range: DateRange,
                  **kwargs) -> ReaderResult:
        """Retrieve data from a web source"""
        _ = kwargs.pop('columns', None)
        freq = kwargs.pop('freq', '1min')

        try:
            df = self.tiingo.get_intra_day(ticker, date_range, freq=freq)
            if df.empty:
                raise RestClientError(f'df retrieved was empty for '
                                      f'ticker: {ticker}.')
        except RestClientError as e:
            logger.warning(f'Unable to retrieve intra day data for '
                           f'ticker: {ticker}. Attempting to get historical '
                           f'data.')
            df = self.tiingo.get_historical_data(ticker, date_range)
            if df.empty:
                return ReaderResult(ticker, successful=False)

        df = utils.rename_bar_cols(df)
        df[utils.TICKER_COL] = ticker
        df[utils.FROM_DB_COL] = False

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index([utils.DATE_COL])
        else:
            df.index.name = utils.DATE_COL

        return ReaderResult(ticker, df)

    def _from_db(self,
                 ticker: str,
                 source: str,
                 date_range: DateRange,
                 **kwargs) -> ReaderResult:
        """
        Try to read data from the DB.

        :param ticker: The ticker to retrieve from the DB.
        :param source: Only used if there there is not enough data in the DB.
        :param start: The start of the range.
        :param end: The end of the range.
        :param filter_data: Passed to the read method.
        :param kwargs: Passed to the read method.
        :return: The data frame.
        :raises: NoDataFoundException if no data is found for the given ticker.
        """
        q = (sa.select([bars])
            .where(sa.and_(bars.c.date >= date_range.start,
                           bars.c.date <= date_range.end + BDay(),
                           bars.c.ticker == ticker)))

        logger.info(f'Checking DB for ticker: {ticker}')
        df = self.reader.df(q)

        if df.empty:
            raise DataAccessError('DataFrame was empty. No data found.')

        df[utils.FROM_DB_COL] = True

        logger.debug(f'Found ticker: {ticker} in DB.')

        db_start = utils.parse_date(df.index.min())
        db_end = utils.parse_date(df.index.max())

        # check that all the requested data is present
        if db_start > date_range.start and date_range.is_trade_day('start'):
            # db has less data than requested

            dt_range_start = None
            try:
                dt_range_start = DateRange(date_range.start, db_start - BDay())
            except ValueError:
                try:
                    dt_range_start = DateRange(date_range.start, db_start)
                except ValueError:
                    logger.warning('Unable to get lower dt_range_start')
            if dt_range_start is not None:
                lower_df = self._from_web(ticker, source, dt_range_start).df
            else:
                lower_df = None
        else:
            lower_df = None

        if db_end < date_range.end and date_range.is_trade_day('end'):
            # db doesn't have as much data than requested
            dt_range_end = DateRange(db_end, date_range.end)
            upper_df = self._from_web(ticker, source, dt_range_end).df
        else:
            upper_df = None

        new_df = _concat_dfs(lower_df, upper_df, df)
        new_df = new_df.sort_index()
        return ReaderResult(ticker, new_df)


def _concat_dfs(lower_df: pd.DataFrame,
                upper_df: pd.DataFrame,
                df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper method to concat the missing data frames, where `df` is the original
    df.
    """

    def do_concat(*args) -> pd.DataFrame:
        dfs = [x for x in args if not x.empty or x is None]
        return pd.concat(dfs, join='inner', axis=0)

    if lower_df is None and upper_df is None:
        return df
    elif lower_df is not None and upper_df is None:
        return do_concat(df, lower_df)
    elif lower_df is None and upper_df is not None:
        return do_concat(df, upper_df)
    elif lower_df is not None and upper_df is not None:
        return do_concat(df, upper_df, lower_df)
    else:
        return df

import datetime as dt
import logging
import queue
from abc import ABCMeta, abstractmethod
from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from arctic.date import DateRange
from arctic.exceptions import NoDataFoundException

import pytech.utils.dt_utils as dt_utils
import pytech.utils.pandas_utils as pd_utils
from pytech.backtest.event import MarketEvent
# import pytech.db.db as db
from pytech.mongo import ARCTIC_STORE, BarStore
from pytech.utils.decorators import memoize


class DataHandler(metaclass=ABCMeta):
    # Type hints
    events: queue.Queue
    tickers: Iterable
    start_date: dt.datetime
    end_date: dt.datetime
    ticker_data: Dict[str, pd.DataFrame]
    lib: BarStore

    CHUNK_SIZE = 'D'

    def __init__(self,
                 events: queue.Queue,
                 tickers: Iterable,
                 start_date: dt.datetime,
                 end_date: dt.datetime):
        """
        All child classes MUST call this constructor.

        :param events: The universal queue.
        :param tickers: An iterable of tickers. This will create the asset
        universe or all the assets that will available to be traded.
        :param start_date: The start of the sim.
        :param end_date: The end of the sim.
        """
        self.logger = logging.getLogger(__name__)
        self.events = events
        self.tickers = tickers
        self.ticker_data = {}
        self.latest_ticker_data = {}
        self.continue_backtest = True
        self.start_date = dt_utils.parse_date(start_date)
        self.end_date = dt_utils.parse_date(end_date)
        self.lib = ARCTIC_STORE['pytech.bars']
        self._populate_ticker_data()

    @abstractmethod
    def get_latest_bar(self, ticker: str):
        """
        Return the latest bar updated.

        :param str ticker: The ticker to retrieve the bar for.
        :return: The latest update bar for the given ticker.
        """
        raise NotImplementedError('Must implement get_latest_bar()')

    @abstractmethod
    def get_latest_bars(self, ticker: str, n: int = 1):
        """
        Returns the last **n** bars from the latest_symbol list,
        or fewer if less bars available.
        :param ticker:
        :param int n: The number of bars.
        :return:
        """
        raise NotImplementedError('Must implement get_latest_bars()')

    @abstractmethod
    def get_latest_bar_dt(self, ticker: str):
        """
        Return the datetime for the last bar for the given ticker.

        :param str ticker: The ticker of the asset.
        :return: The datetime of the last bar for the given asset
        :rtype: dt.datetime
        """
        raise NotImplementedError('Must implement get_latest_bar_dt()')

    @abstractmethod
    def get_latest_bar_value(self, ticker: str, val_type, n=1):
        """
        Return the last **n** bars from the latest_symbol list.

        :param ticker:
        :param val_type:
        :param n:
        :return:
        """
        raise NotImplementedError('Must implement get_latest_bar_value()')

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest symbol structure for all symbols
        in the symbol list.
        """
        raise NotImplementedError('Must implement update_bars()')

    @abstractmethod
    def _populate_ticker_data(self):
        """
        Populate the ticker_data dict with a pandas OHLCV
        df as the value and the ticker as the key.

        This will get called on ``__init__`` and is **NOT** intended to ever
        be called directly by child classes.
        """
        raise NotImplementedError('Must implement _populate_ticker_data()')


class YahooDataHandler(DataHandler):
    DATA_SOURCE = 'yahoo'

    def __init__(self,
                 events: queue.Queue,
                 tickers: Iterable,
                 start_date: dt.datetime,
                 end_date: dt.datetime):
        super().__init__(events, tickers, start_date, end_date)

    def _populate_ticker_data(self):
        """
        Populate the ticker_data dict with a pandas OHLCV
        df as the value and the ticker as the key.
        """
        comb_index = None
        # only create the DateRange object once.
        date_range = DateRange(start=self.start_date, end=self.end_date)

        for t in self.tickers:
            self.ticker_data[t] = self._get_ticker_df(t, date_range)

            if comb_index is None:
                comb_index = self.ticker_data[t].index
            else:
                comb_index.union(self.ticker_data[t].index)

            self.latest_ticker_data[t] = []

        for t in self.tickers:
            self.ticker_data[t] = (self.ticker_data[t]
                                   .reindex(index=comb_index, method='pad')
                                   .iterrows())

    def _get_data_from_web(self, ticker: str,
                           date_range: DateRange = None) -> pd.DataFrame:
        """Make the network call and write the new df to the DB."""
        if date_range is not None:
            start = date_range.start
            end = date_range.end
        else:
            start = self.start_date
            end = self.end_date

        df = web.DataReader(ticker, self.DATA_SOURCE, start, end)
        # df = web.get_data_yahoo(ticker, start=start, end=end)

        # TODO: generalize this.
        df = pd_utils.rename_bar_cols(df)
        # set index to the date
        df = df.set_index([pd_utils.DATE_COL])
        self.lib.write(ticker, df, self.CHUNK_SIZE)
        return df

    @memoize
    def make_agg_df(self, col: str = pd_utils.ADJ_CLOSE_COL,
                    market_ticker: Union[str, None] = 'SPY') -> pd.DataFrame:
        """
        Make a df that contains all of the ticker data and write it the db.

        This is used to do analysis like correlation, so a market ticker should
        be added.

        :param col: The column to use to create the aggregate DF.
        :param market_ticker: The ticker that will be used to represent the
        market. If None is passed then no market_ticker will be used.
        :return: The aggregate data frame.
        """
        date_range = DateRange(start=self.start_date, end=self.end_date)
        agg_df = pd.DataFrame()

        for t in self.tickers:
            try:
                temp_df = self.lib.read(t, chunk_range=date_range)
            except NoDataFoundException:
                temp_df = self._get_data_from_web(t)

            agg_df[t] = temp_df[col]

        if market_ticker is not None and market_ticker not in self.tickers:
            temp = self._get_ticker_df(market_ticker, date_range)
            agg_df[market_ticker] = temp[col]

        return agg_df

    def _get_ticker_df(self, ticker: str,
                       chunk_range: DateRange) -> pd.DataFrame:
        """
        Helper method to get a :class:``pd.DataFrame`` for a ticker from either
        the database or the web if the requested ticker is not found in the db.

        :param ticker: The ticker to get a df for.
        :param chunk_range: The time range to get the data for.
        :return: The data frame.
        """
        try:
            return self.lib.read(ticker, chunk_range=chunk_range)
        except NoDataFoundException:
            return self._get_data_from_web(ticker, date_range=chunk_range)

    def _get_new_bar(self, ticker: str):
        """
        Get the latest bar from the data feed and return it as a tuple.

        :return: bar
        """
        for bar in self.ticker_data[ticker]:
            yield bar

    def get_latest_bar(self, ticker: str):

        try:
            bars_list = self.latest_ticker_data[ticker]
        except KeyError:
            self.logger.exception(
                    f'{ticker} is not available in the given data set.')
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, ticker: str, n: int = 1):
        """
        Returns the last ``n`` bars from the latest_ticker_data.
        If there is less than ``n`` bars available then n-k is returned.

        :param str ticker: The ticker of the asset for which the bars are
        needed.
        :param int n: The number of bars to return.
        (default: 1)
        :return: A list of bars.
        """
        try:
            bars_list = self.latest_ticker_data[ticker]
        except KeyError:
            self.logger.exception(
                    f'Could not find {ticker} in latest_ticker_data')
            raise
        else:
            return bars_list[-n:]

    def get_latest_bar_dt(self, ticker) -> dt.datetime:
        try:
            bars_list = self.latest_ticker_data[ticker]
        except KeyError:
            self.logger.exception(
                    f'Could not find {ticker} in latest_ticker_data')
            raise
        else:
            return dt_utils.parse_date(bars_list[-1].name)

    def get_latest_bar_value(self, ticker, val_type, n=1):
        """
        Get the last ``n`` bars but return a series containing only the
        ``val_type`` requested.

        :param str ticker: The ticker of the asset for which the bars are
        needed.
        :param val_type:
        :param n:
        :return:
        """
        try:
            bars_list = self.get_latest_bars(ticker, n)
        except KeyError:
            self.logger.exception(
                    f'Could not find {ticker} in latest_ticker_data')
            raise
        else:
            return np.array([getattr(bar, val_type) for bar in bars_list])

    def update_bars(self):
        for ticker in self.tickers:
            try:
                bar = next(self._get_new_bar(ticker))
                # bar is a tuple and we only care about the 2nd value in it.
                bar = bar[1]
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_ticker_data[ticker].append(bar)

        self.events.put(MarketEvent())

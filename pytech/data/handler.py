import datetime as dt
import numpy as np
import logging
from abc import ABCMeta, abstractmethod
from pytech.backtest.event import MarketEvent
import pytech.utils.pandas_utils as pd_utils
import pandas_datareader.data as web


class DataHandler(metaclass=ABCMeta):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_latest_bar(self, ticker):
        """
        Return the latest bar updated.
        :param str ticker: The ticker to retrieve the bar for.
        :return: The latest update bar for the given ticker.
        """

        raise NotImplementedError('Must implement get_latest_bar()')

    @abstractmethod
    def get_latest_bars(self, ticker, n=1):
        """
        Returns the last **n** bars from the latest_symbol list, or fewer if less bars available.
        :param ticker:
        :param int n: The number of bars.
        :return:
        """

        raise NotImplementedError('Must implement get_latest_bars()')

    @abstractmethod
    def get_latest_bar_dt(self, ticker):
        """
        Return the datetime for the last bar for the given ticker.

        :param str ticker: The ticker of the asset.
        :return: The datetime of the last bar for the given asset
        :rtype: dt.datetime
        """

        raise NotImplementedError('Must implement get_latest_bar_dt()')

    @abstractmethod
    def get_latest_bars_values(self, ticker, val_type, n=1):
        """
        Return the last **n** bars from the latest_symbol list.

        :param ticker:
        :param val_type:
        :param n:
        :return:
        """

        raise NotImplementedError('Must implement get_latest_bars_values()')

    @abstractmethod
    def update_bars(self):
        """Pushes the latest bar to the latest symbol structure for all symbols in the symbol list."""

        raise NotImplementedError('Must implement update_bars()')


class YahooDataHandler(DataHandler):

    DATA_SOURCE = 'yahoo'

    def __init__(self, events, ticker_list, start_date, end_date):

        super().__init__()
        self.events = events
        self.ticker_list = ticker_list
        self.ticker_data = {}
        self.latest_ticker_data = {}
        self.continue_backtest = True
        self._get_ohlcvs(start_date, end_date)

    def _get_ohlcvs(self, start_date, end_date):
        """Populate the ticker_data dict with a pandas OHLCV df as the value and the ticker as the key."""

        comb_index = None

        for t in self.ticker_list:
            self.ticker_data[t] = web.DataReader(t, data_source=self.DATA_SOURCE, start=start_date, end=end_date)

            if comb_index is None:
                comb_index = self.ticker_data[t].index
            else:
                comb_index.union(self.ticker_data[t].index)

            self.latest_ticker_data[t] = []

        for t in self.ticker_list:
            self.ticker_data[t] = self.ticker_data[t].reindex(index=comb_index, method='pad').iterrows()
            self.ticker_data[t] = pd_utils.rename_yahoo_ohlcv_cols(self.ticker_data[t])

    def _get_new_bar(self, ticker):
        """
        Get the latest bar from the data feed and return it as a tuple.
        :return: tuple(ticker, dt, open, high, low, close, volume)
        """

        for bar in self.ticker_data[ticker]:
            yield bar

    def get_latest_bar(self, ticker):

        try:
            bars_list = self.latest_ticker_data[ticker]
        except KeyError:
            self.logger.exception('{} is not available in the given data set.')
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, ticker, n=1):

        try:
            bars_list = self.latest_ticker_data[ticker]
        except KeyError:
            self.logger.exception('Could not find {ticker} in latest_ticker_data'.format(ticker=ticker))
            raise
        else:
            return bars_list[-n:]

    def get_latest_bar_dt(self, ticker):

        try:
            bars_list = self.latest_ticker_data[ticker]
        except KeyError:
            self.logger.exception('Could not find {ticker} in latest_ticker_data'.format(ticker=ticker))
            raise
        else:
            return bars_list[-1][pd_utils.DATE_COL]

    def get_latest_bars_values(self, ticker, val_type, n=1):

        try:
            bars_list = self.get_latest_bars(ticker, n)
        except KeyError:
            self.logger.exception('Could not find {ticker} in latest_ticker_data'.format(ticker=ticker))
            raise
        else:
            return np.array([getattr(bar[1], val_type) for bar in bars_list])

    def update_bars(self):

        for ticker in self.ticker_list:
            try:
                bar = next(self._get_new_bar(ticker))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_ticker_data[ticker].append(bar)

        self.events.put(MarketEvent())

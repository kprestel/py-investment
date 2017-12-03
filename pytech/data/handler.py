import datetime as dt
import logging
import queue
from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import (
    Dict,
    Iterable,
    List,
    Union,
)

import numpy as np
import pandas as pd

import pytech.utils as utils
from pytech.backtest.event import MarketEvent
from pytech.decorators import (
    lazy_property,
    memoize,
)
from pytech.utils import DateRange
from ..data import BarReader


class DataHandler(metaclass=ABCMeta):
    CHUNK_SIZE = 'D'

    def __init__(self,
                 events: queue.Queue,
                 tickers: Iterable[str],
                 date_range: DateRange,
                 asset_lib_name: str = 'pytech.bars',
                 market_lib_name: str = 'pytech.market') -> None:
        """
        All child classes MUST call this constructor.

        :param events: The universal queue.
        :param tickers: An iterable of tickers. This will create the asset
        universe or all the assets that will available to be traded.
        :param start_date: The start of the sim.
        :param end_date: The end of the sim.
        :param asset_lib_name: The name of the mongo library where asset
            bars are stored. Defaults to *pytech.bars*
        :param market_lib_name: The name of the mongo library where market
            bars are stored. Defaults to *pytech.market*
        """
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.events: queue.Queue = events
        self.tickers: List[str] = []
        self.tickers.extend(tickers)
        # self._ticker_data = {}
        self.latest_ticker_data = {}
        self.continue_backtest: bool = True
        self.date_range: DateRange = date_range or DateRange()
        self.asset_lib_name: str = asset_lib_name
        self.market_lib_name: str = market_lib_name
        self.asset_reader: BarReader = BarReader()
        self.mkt_reader: BarReader = BarReader()

    @lazy_property
    def ticker_data(self):
        return self._populate_ticker_data()

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
    def latest_bar_value(self, ticker: str, val_type, n=1):
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


class Bars(DataHandler):
    def __init__(self,
                 events: queue.Queue,
                 tickers: Iterable,
                 date_range: DateRange,
                 source: str = 'yahoo',
                 asset_lib_name: str = 'pytech.bars',
                 market_lib_name: str = 'pytech.market'):
        self.source = source
        super().__init__(events=events,
                         tickers=tickers,
                         date_range=date_range,
                         asset_lib_name=asset_lib_name,
                         market_lib_name=market_lib_name)

    def _populate_ticker_data(self) -> Dict[str, Iterable[pd.Series]]:
        """
        Populate the ticker_data dict with a pandas OHLCV
        df as the value and the ticker as the key.
        """
        comb_index = None
        df_dict = self._get_data()
        out = {}

        for t in self.tickers:
            out[t] = df_dict[t]

            # TODO needed?
            if comb_index is None:
                comb_index = out[t].index.get_level_values(utils.DATE_COL)
            else:
                comb_index.union(out[t].index.get_level_values(utils.DATE_COL))

            self.latest_ticker_data[t] = []

        for t in self.tickers:
            out[t] = out[t].iterrows()
            # self.ticker_data[t] = (self.ticker_data[t].iterrows())
        return out

    @memoize
    def make_agg_df(self, col: str = utils.CLOSE_COL,
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
        agg_df = pd.DataFrame()
        df_dict = self._get_data()

        if market_ticker is not None and market_ticker not in self.tickers:
            # get the market data if it has not already been fetched
            market_df: pd.DataFrame = self.mkt_reader.get_data(market_ticker,
                                                               columns=col,
                                                               date_range=self.date_range)
            if agg_df.empty:
                agg_df = pd.DataFrame(market_df[col].values,
                                      index=market_df.index,
                                      columns=[market_ticker])

        for t in self.tickers:
            temp_df: pd.DataFrame = df_dict[t].copy()
            temp_df = pd.DataFrame(temp_df[col].values,
                                   index=temp_df.index,
                                   columns=[t])
            if agg_df.empty:
                agg_df = df_dict[t][[col, utils.TICKER_COL]]
            else:
                agg_df = pd.merge(agg_df, temp_df, left_index=True,
                                  right_index=True, how='outer')

        return agg_df

    @memoize
    def _get_data(self,
                  tickers: Iterable[str] = None,
                  **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Get the data.

        :param tickers: any extra tickers to get data for.
        :return:
        """
        if tickers is not None:
            tickers = self.tickers
            tickers.extend(tickers)
        else:
            tickers = self.tickers

        return self.asset_reader.get_data(tickers,
                                          self.source,
                                          self.date_range,
                                          **kwargs)

    def _get_new_bar(self, ticker: str):
        """
        Get the latest bar from the data feed and return it as a tuple.

        :return: bar
        """
        # yield from self.ticker_data
        for bar in self.ticker_data[ticker]:
            yield bar

    def get_latest_bar(self, ticker: str) -> pd.DataFrame:
        """Return the latest bar."""
        try:
            bars_list = self.latest_ticker_data[ticker]
        except KeyError:
            raise KeyError(f'{ticker} is not available in the given data set.')
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
            raise KeyError(f'Could not find {ticker} in latest_ticker_data')
        else:
            return bars_list[-n:]

    def get_latest_bar_dt(self, ticker) -> dt.datetime:
        try:
            bars_list = self.latest_ticker_data[ticker]
        except KeyError:
            raise KeyError(f'Could not find {ticker} in latest_ticker_data')
        else:
            return utils.parse_date(bars_list[-1].name)

    def latest_bar_value(self, ticker, col, n=1):
        """
        Get the last ``n`` bars but return an array containing only the
        ``val_type`` requested.

        :param str ticker: The ticker of the asset for which the bars are
        needed.
        :param col: the column to get the values for.
        :param n: the number of bars of data to get.
        :return:
        """
        try:
            bars_list = self.get_latest_bars(ticker, n)
        except KeyError:
            raise KeyError(f'Could not find {ticker} in latest_ticker_data')

        # ADJ_CLOSE is only in data from yahoo
        if hasattr(bars_list[-1], col):
            return np.array([getattr(bar, col) for bar in bars_list])
        elif col == utils.ADJ_CLOSE_COL and hasattr(bars_list[-1],
                                                    utils.CLOSE_COL):
            self.logger.warning(f'{col} was requested but was not found, '
                                f'using {utils.CLOSE_COL} instead.')
            return np.array(
                [getattr(bar, utils.CLOSE_COL) for bar in bars_list])
        else:
            raise ValueError(f'{col} does not exist in bars.')

    def update_bars(self) -> None:
        for ticker in self.tickers:
            bar = next(self._get_new_bar(ticker))
            # bar is a tuple and we only care about the 2nd value in it.
            bar = bar[1]
            if bar is not None:
                self.latest_ticker_data[ticker].append(bar)
        self.events.put(MarketEvent())

import os
from dateutil.tz import tzutc
from dateutil.parser import parse
import queue
from typing import Dict
from unittest.mock import MagicMock as Mock

import pandas as pd
import pytest

import pytech.trading.blotter as b
from pytech.fin.portfolio.handler import BasicSignalHandler
from pytech import TEST_DATA_DIR
from pytech.data.handler import Bars
from pytech.data import BarReader
from pytech.fin.asset.asset import Stock
from pytech.fin.portfolio import BasicPortfolio
from pytech.mongo import ARCTIC_STORE
from pytech.trading.controls import MaxOrderCount
import pytech.utils as utils

lib = ARCTIC_STORE['pytech.bars']


@pytest.fixture(scope='session')
def start_date():
    return '2016-03-10'


@pytest.fixture(scope='session')
def end_date():
    return '2017-06-09'

@pytest.fixture(scope='session')
def _ticker_df_cache():
    """
    Create a dictionary of dfs so we only have to go to disk once.
    """
    df_cache = {}
    for f in os.listdir(TEST_DATA_DIR):
        df = pd.read_csv(os.path.join(TEST_DATA_DIR, f),
                         index_col=utils.DATE_COL,
                         parse_dates=['date'])
        df.index.name = utils.DATE_COL
        df_cache[os.path.splitext(os.path.basename(f))[0]] = df
    return df_cache


# @pytest.fixture(autouse=True)
def write_ref_csv(monkeypatch, start_date, end_date):
    """
    This is a utils fixture that shouldn't be used unless generating
    reference data.
    """

    def to_csv(bar_reader: BarReader, tickers,
               source='google',
               start=start_date,
               end=end_date,
               check_db=True,
               filter_data=True,
               **kwargs):
        start = utils.parse_date(start)
        end = utils.parse_date(end)
        _ = kwargs.pop('columns', None)
        if isinstance(tickers, str):
            df = bar_reader._single_get_data(tickers,
                                             source,
                                             start,
                                             end,
                                             check_db,
                                             filter_data,
                                             **kwargs)
            df.df.to_csv(f'{TEST_DATA_DIR}{os.sep}{tickers}.csv')
            return df.df
        else:
            if isinstance(tickers, pd.DataFrame):
                tickers = tickers.index
            df = bar_reader._mult_tickers_get_data(tickers,
                                                   source,
                                                   start,
                                                   end,
                                                   check_db,
                                                   filter_data,
                                                   **kwargs)
            for ticker, df_ in df.items():
                df_.to_csv(f'{TEST_DATA_DIR}{os.sep}{ticker}.csv')
            return df

    monkeypatch.setattr(BarReader, 'get_data', to_csv)


@pytest.fixture(autouse=True)
def no_db(monkeypatch, _ticker_df_cache: Dict[str, pd.DataFrame]):
    """Don't make any database calls. Read all data from `TEST_DATA_DIR`"""

    def patch_get_data(bar_reader, tickers, *args, **kwargs):
        if isinstance(tickers, str):
            return _ticker_df_cache[tickers]

        dfs = {}
        for ticker in tickers:
            df = _ticker_df_cache[ticker]
            dfs[ticker] = df

        return dfs

    monkeypatch.setattr(BarReader, 'get_data', patch_get_data)


@pytest.fixture(scope='module')
def aapl_df():
    """Returns a OHLCV df for Apple."""
    return pd.read_csv(f'{TEST_DATA_DIR}{os.sep}AAPL.csv')

def date_utc(s):
    return parse(s, tzinfos=tzutc)

@pytest.fixture(scope='module')
def cvs_df():
    return pd.read_csv(f'{TEST_DATA_DIR}{os.sep}CVS.csv')
    # return pd.read_csv(f'{TEST_DATA_DIR}{os.sep}CVS.csv', parse_dates=['date'],
    #                    date_parser=date_utc)


def get_test_csv_path(ticker):
    """Return the path to the test CSV file"""
    return f'{TEST_DATA_DIR}{os.sep}{ticker}.csv'


@pytest.fixture(scope='session')
def ticker_list():
    return {'AAPL', 'MSFT', 'CVS', 'FB'}


@pytest.fixture(scope='session')
def events():
    return queue.Queue()


@pytest.fixture()
def mock_portfolio():
    """A mock portfolio that does nothing but be a mock."""
    return Mock(spec=BasicPortfolio)


@pytest.fixture()
def bars(events, ticker_list, start_date, end_date):
    """Create a default :class:`YahooDataHandler`"""
    bars = Bars(events, ticker_list, start_date, end_date)
    bars.update_bars()
    return bars


@pytest.fixture()
def blotter(events, bars):
    return b.Blotter(events, bars=bars)


@pytest.fixture()
def populated_blotter(blotter: b.Blotter, mock_portfolio, start_date):
    """Populate the blot and return it."""
    blotter.controls.append(MaxOrderCount(True, 10))
    blotter.current_dt = start_date

    blotter.place_order(mock_portfolio, 'AAPL', 50, 'BUY', 'LIMIT',
                        limit_price=100.10,
                        order_id='one')
    blotter.place_order(mock_portfolio, 'AAPL', 50, 'BUY', 'LIMIT',
                        limit_price=98.10,
                        order_id='two')
    blotter.place_order(mock_portfolio, 'MSFT', 50, 'SELL', 'LIMIT',
                        limit_price=93.10,
                        order_id='three')
    blotter.place_order(mock_portfolio, 'FB', 50, 'SELL', 'LIMIT',
                        limit_price=105.10,
                        order_id='four')

    return blotter


@pytest.fixture()
def basic_portfolio(events, bars, start_date, populated_blotter):
    """Return a BasicPortfolio to be used in testing."""
    populated_blotter.bars = bars
    return BasicPortfolio(bars, events, start_date,
                          populated_blotter)


@pytest.fixture()
def empty_portfolio(events, start_date, blotter):
    return BasicPortfolio(blotter.bars, events, start_date, blotter)


@pytest.fixture()
def basic_signal_handler(basic_portfolio):
    """Return a BasicSignalHandler to be used in testing."""
    return BasicSignalHandler()


@pytest.fixture()
def aapl(start_date, end_date):
    return Stock('AAPL', start_date, end_date)


@pytest.fixture()
def fb(start_date, end_date):
    return Stock('FB', start_date, end_date)

import os
import queue
from unittest.mock import MagicMock as Mock

import pandas as pd
import pytest

import pytech.trading.blotter as b
from pytech.fin.portfolio.handler import BasicSignalHandler
from pytech import TEST_DATA_DIR
from pytech.data.handler import Bars
from pytech.fin.asset.asset import Stock
from pytech.fin.portfolio import BasicPortfolio
from pytech.mongo import ARCTIC_STORE
from pytech.trading.controls import MaxOrderCount

lib = ARCTIC_STORE['pytech.bars']


@pytest.fixture(scope='session')
def start_date():
    return '2016-03-10'


@pytest.fixture(scope='session')
def end_date():
    return '2017-06-09'


# @pytest.fixture(autouse=True)
def no_requests(monkeypatch):
    """Prevent making requests to yahoo to speed up testing"""

    def patch_requests(ticker, data_source, start, end):
        # return lib.read(ticker)
        # return db.read(ticker)
        return pd.read_csv(get_test_csv_path(ticker), parse_dates=['Date'])

    monkeypatch.setattr('pandas_datareader.data.DataReader', patch_requests)


@pytest.fixture(scope='module')
def aapl_df():
    """Returns a OHLCV df for Apple."""
    return lib.read('AAPL')


def get_test_csv_path(ticker):
    """Return the path to the test CSV file"""
    return TEST_DATA_DIR + os.sep + '{}.csv'.format(ticker)


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
def basic_signal_handler(basic_portfolio):
    """Return a BasicSignalHandler to be used in testing."""
    return BasicSignalHandler()


@pytest.fixture()
def aapl(start_date, end_date):
    return Stock('AAPL', start_date, end_date)


@pytest.fixture()
def fb(start_date, end_date):
    return Stock('FB', start_date, end_date)

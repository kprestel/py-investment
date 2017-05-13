import os
import queue

import pandas as pd
import pytest

import pytech.trading.blotter as b
from pytech import TEST_DATA_DIR
from pytech.data.handler import YahooDataHandler
from pytech.fin.portfolio import BasicPortfolio


@pytest.fixture()
def start_date():
    return '2016-03-10'


@pytest.fixture()
def end_date():
    return '2017-01-01'


@pytest.fixture(autouse=True)
def no_requests(monkeypatch):
    """Prevent making requests to yahoo to speed up testing"""

    def patch_requests(ticker, data_source, start, end):
        return pd.read_csv(get_test_csv_path(ticker))

    monkeypatch.setattr('pandas_datareader.data.DataReader', patch_requests)


def get_test_csv_path(ticker):
    """Return the path to the test CSV file"""

    return TEST_DATA_DIR + os.sep + '{}.csv'.format(ticker)


@pytest.fixture()
def ticker_list():
    return {'AAPL', 'MSFT', 'FB', 'IBM', 'SPY', 'GOOG', 'AMZN', 'SKX', 'COST',
            'CVS', 'EBAY', 'INTC', 'NKE', 'PYPL'}


@pytest.fixture()
def events():
    return queue.Queue()


@pytest.fixture()
def blotter(events):
    return b.Blotter(events)


@pytest.fixture()
def populated_blotter(blotter):
    """Populate the blot and return it."""

    blotter.place_order('AAPL', 'BUY', 'LIMIT', 50, limit_price=100.10,
                        order_id='one')
    blotter.place_order('AAPL', 'BUY', 'LIMIT', 50, limit_price=98.10,
                        order_id='two')
    blotter.place_order('MSFT', 'SELL', 'LIMIT', 50, limit_price=93.10,
                        order_id='three')
    blotter.place_order('FB', 'SELL', 'LIMIT', 50, limit_price=105.10,
                        order_id='four')

    return blotter


@pytest.fixture()
def yahoo_data_handler(events, ticker_list, start_date, end_date):
    """Create a default :class:`YahooDataHandler`"""
    bars = YahooDataHandler(events, ticker_list, start_date, end_date)
    bars.update_bars()
    return bars


@pytest.fixture()
def basic_portfolio(events, yahoo_data_handler, start_date, populated_blotter):
    populated_blotter.bars = yahoo_data_handler
    return BasicPortfolio(yahoo_data_handler, events, start_date,
                          populated_blotter)

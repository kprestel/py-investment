import os
import pandas as pd
import queue

import pytest
import queuelib.queue

from pytech import TEST_DATA_DIR
import pytech.trading.blotter as b

from pytech.fin.asset import Fundamental, Stock

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
    return ['AAPL', 'MSFT', 'FB', 'IBM']

@pytest.fixture()
def events():
    return queue.Queue()


@pytest.fixture()
def blotter(events):
    return b.Blotter(events)


@pytest.fixture()
def populated_blotter(blotter):
    """Populate the blot and return it."""

    blotter.place_order('AAPL', 'BUY', 'LIMIT', 50, limit_price=100.10, order_id='one')
    blotter.place_order('AAPL', 'BUY', 'LIMIT', 50, limit_price=98.10, order_id='two')
    blotter.place_order('MSFT', 'SELL', 'LIMIT', 50, limit_price=93.10, order_id='three')
    blotter.place_order('FB', 'SELL', 'LIMIT', 50, limit_price=105.10, order_id='four')

    return blotter

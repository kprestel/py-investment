# noinspection PyUnresolvedReferences
import datetime as dt
import pytest
from pytech.sources.tiingo import TiingoClient
from utils import DateRange


class TestTiingoClient(object):
    client = TiingoClient()

    def test_get_ticker_metadata(self):
        resp = self.client.get_ticker_metadata('AAPL')
        assert resp is not None

    def test_get_ticker_prices(self, date_range):
        resp = self.client.get_historical_data('AAPL', date_range=date_range)
        assert resp is not None

    def test_get_intra_day(self):
        date_range = DateRange(dt.datetime(2017, 11, 30),
                               dt.datetime(2017, 12, 1))
        resp = self.client.get_intra_day('AAPL', date_range=date_range)
        assert resp is not None

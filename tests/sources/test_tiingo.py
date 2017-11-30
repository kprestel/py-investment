# noinspection PyUnresolvedReferences
import pytest
from pytech.sources.tiingo import TiingoClient

class TestTiingoClient(object):

    client = TiingoClient()

    def test_get_ticker_metadata(self):
        resp = self.client.get_ticker_metadata('AAPL')
        assert resp is not None

    def test_get_ticker_prices(self, date_range):
        resp = self.client.get_ticker_prices('AAPL', date_range=date_range)
        assert resp is not None

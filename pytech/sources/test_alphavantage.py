# noinspection PyUnresolvedReferences
import pytest

from sources.alphavantage import AlphaVantageClient


class TestAlphaVantage(object):
    client = AlphaVantageClient()

    def test_get_intra_day(self):
        df = self.client.get_intra_day('FB')
        assert df is not None

    def test_get_daily(self):
        df = self.client.get_daily('FB')
        assert df is not None

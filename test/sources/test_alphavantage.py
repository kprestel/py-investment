# noinspection PyUnresolvedReferences
import pytest
import vcr

from sources.alphavantage import AlphaVantageClient
from utils import DateRange


class TestAlphaVantage(object):
    client = AlphaVantageClient()

    @pytest.mark.vcr
    def test_get_intra_day(self):
        date_range = DateRange('2017-11-30', '2017-12-01')
        df = self.client.get_intra_day('FB', date_range)
        assert df is not None

    def test_get_daily(self):
        df = self.client.get_daily('FB')
        assert df is not None

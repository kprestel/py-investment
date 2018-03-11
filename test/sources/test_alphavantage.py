# noinspection PyUnresolvedReferences
import pytest

from pytech.sources.alphavantage import AlphaVantageClient
from pytech.utils import DateRange


class TestAlphaVantage(object):
    client = AlphaVantageClient()

    @pytest.mark.vcr
    def test_get_intra_day(self):
        date_range = DateRange('2017-11-30', '2017-12-01')
        df = self.client.get_intra_day('FB', date_range)
        assert df is not None

    @pytest.mark.vcr
    def test_get_historical_daily(self, date_range):
        df = self.client.get_historical_data('FB',
                                             date_range,
                                             'Daily',
                                             adjusted=True)
        assert df is not None

    @pytest.mark.vcr
    def test_get_historical_weekly(self, date_range):
        df = self.client.get_historical_data('FB',
                                             date_range,
                                             'Weekly',
                                             adjusted=True)
        assert df is not None

    @pytest.mark.vcr
    def test_get_historical_monthly(self, date_range):
        df = self.client.get_historical_data('FB',
                                             date_range,
                                             'Monthly',
                                             adjusted=True)
        assert df is not None

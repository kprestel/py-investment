# noinspection PyUnresolvedReferences
import datetime as dt
import pytest
from pytech.sources.tiingo import TiingoClient
from pytech.utils import DateRange


class TestTiingoClient(object):
    client = TiingoClient()

    @pytest.mark.vcr
    def test_get_ticker_metadata(self):
        resp = self.client.get_ticker_metadata('AAPL')
        assert resp is not None

    @pytest.mark.vcr
    def test_get_historical_data_daily_adj(self, date_range):
        resp = self.client.get_historical_data('AAPL',
                                               freq='daily',
                                               date_range=date_range)
        assert resp is not None

    @pytest.mark.vcr
    def test_get_intra_day(self):
        date_range = DateRange(dt.datetime(2017, 11, 30),
                               dt.datetime(2017, 12, 1))
        resp = self.client.get_intra_day('AAPL', date_range=date_range)
        assert resp is not None

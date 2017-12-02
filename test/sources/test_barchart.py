import pytest

from pytech.sources.barchart import BarChartClient

class TestBarChartClient(object):
    client = BarChartClient()

    @pytest.mark.skip
    def test_quote(self):
        resp = self.client.quote(('AAPL', 'GOOG'))
        assert resp is not None

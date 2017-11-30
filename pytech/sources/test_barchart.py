# noinspection PyUnresolvedReferences
import pytest

from pytech.sources.barchart import BarChartClient

class TestBarChartClient(object):
    client = BarChartClient()

    def test_quote(self):
        resp = self.client.quote(('AAPL', 'GOOG'))
        assert resp is not None

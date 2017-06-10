# noinspection PyUnresolvedReferences
import pytest

from pytech.fin.asset.asset import Stock


class TestStock(object):
    def test_init(self, start_date, end_date):
        test = Stock('GOOG', start_date, end_date)
        assert test is not None

    def test_calculate_beta(self, start_date, end_date):
        test = Stock('FB', start_date, end_date)
        beta = test.rolling_beta()
        assert 1.565255932552136 == pytest.approx(beta.iloc[-1])

# noinspection PyUnresolvedReferences
import pytest

from pytech.fin.asset.asset import Stock


class TestStock(object):

    # TODO parametrize these, or the whole class.

    def test_calculate_beta(self, fb: Stock):
        # noinspection PyArgumentEqualDefault
        beta = fb.rolling_beta(window=30)
        assert 1.565255932552136 == pytest.approx(beta.iloc[-1])

    def test_avg_return(self, fb: Stock):
        ret = fb.avg_return()
        assert 0.279689869414 == pytest.approx(ret)

    def test_cagr(self, fb: Stock):
        cagr = fb.cagr()
        assert 0.284574337866 == pytest.approx(cagr)


# noinspection PyUnresolvedReferences
import pytest

from pytech.fin.asset.asset import Stock


class TestStock(object):

    # TODO parametrize these, or the whole class.

    def test_calculate_beta(self, fb: Stock):
        # noinspection PyArgumentEqualDefault
        beta = fb.rolling_beta(window=30)
        expected = 1.4993071589622033
        assert expected == pytest.approx(beta.iloc[-1])

    def test_avg_return(self, fb: Stock):
        ret = fb.avg_return()
        expected = 0.28327454495682713
        assert expected == pytest.approx(ret)

    def test_cagr(self, fb: Stock):
        cagr = fb.cagr()
        assert 0.284574337866 == pytest.approx(cagr)


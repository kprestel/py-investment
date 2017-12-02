import pytest
from pytech.fin.analysis.fixed import TVM


class TestTVM(object):

    def test_calc_rate(self):
        """
        Calculate the interest rate and then use it to solve for the
        PV and ensure that it is correct.
        """
        expected_pv = 80.0
        tvm = TVM(periods=10*2, pmt=6/2, pv=-expected_pv, fv=100)
        ir = 2 * tvm.calc_rate()
        tvm.rate = ir
        pv = tvm.calc_pv()
        assert pv == expected_pv

        tvm = TVM(periods=360, rate=1.21, fv=100)
        print(tvm.calc_pv())


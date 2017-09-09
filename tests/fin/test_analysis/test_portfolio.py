# noinspection PyUnresolvedReferences
import pytest

from pytech.fin.analysis.portfolio import EfficientFrontier


class TestEfficientFrontier(object):
    def test_standard_frontier(self):
        frontier = EfficientFrontier()
        result = frontier()
        print(str(result))
        # result.plot()

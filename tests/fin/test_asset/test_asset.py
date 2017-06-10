# noinspection PyUnresolvedReferences
import pytest
from pytech.fin.asset.asset import Stock
import numpy as np


class TestStock(object):

    def test_init(self, start_date, end_date):
        test = Stock('GOOG', start_date, end_date)
        assert test is not None

    def test_calculate_beta(self, start_date, end_date):
        test = Stock('FB', start_date, end_date)
        beta = test.calculate_beta()
        assert 0.8821683029866113 == pytest.approx(beta)




# noinspection PyUnresolvedReferences
import pytest

from fin.asset.asset import Stock
import pytech.fin.analysis.random as rand


def test_monte_carlo(fb: Stock):
    cagr = fb.cagr()
    vol = fb.std()
    last_price = fb.last_price()
    print(last_price)
    trading_days = 252

    output = rand.monte_carlo(cagr, vol, trading_days, last_price)
    print(output)



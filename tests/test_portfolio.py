import pytest
import pytech.fin.portfolio as portfolio


class TestPortfolio(object):

    def test_check_liquidity(self):

        test_portfolio = portfolio.Portfolio(starting_cash=100)

        liquid = test_portfolio.check_liquidity(5.0, 100.0, 2)
        assert liquid is False

        liquid = test_portfolio.check_liquidity(5.0, 100.0, -2)
        assert liquid is True

        liquid = test_portfolio.check_liquidity(5.0, 1.0, 2)
        assert liquid is True

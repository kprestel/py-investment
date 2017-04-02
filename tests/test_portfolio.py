import pytest
from pytech.fin.portfolio import Portfolio, BasicPortfolio


class TestPortfolio(object):

    def test_check_liquidity(self):

        test_portfolio = Portfolio(starting_cash=100)

        liquid = test_portfolio.check_liquidity(100.0, 2)
        assert liquid is False

        liquid = test_portfolio.check_liquidity(100.0, -2)
        assert liquid is True

        liquid = test_portfolio.check_liquidity(1.0, 2)
        assert liquid is True


class TestBasicPortfolio(object):

    def test_constructor(self, yahoo_data_handler, events, blotter):
        test_portfolio = BasicPortfolio(yahoo_data_handler, events,
                                        '2016-03-10', blotter)
        assert test_portfolio is not None

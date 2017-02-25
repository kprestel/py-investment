import pytest
import pytech.trading.blotter as blotter
import pytech.db.finders as finders
import pytech.fin.portfolio as portfolio
from pytech.utils.exceptions import NotAPortfolioError, NotAFinderError


class TestBlotter(object):

    def test_blotter_constructor(self):
        test_blotter = blotter.Blotter()

        assert isinstance(test_blotter.portfolio, portfolio.Portfolio)
        assert isinstance(test_blotter.asset_finder, finders.AssetFinder)

        test_portfolio = portfolio.Portfolio()
        test_finder = finders.AssetFinder()

        test_blotter = blotter.Blotter(portfolio=test_portfolio)
        assert isinstance(test_blotter.portfolio, portfolio.Portfolio)
        assert isinstance(test_blotter.asset_finder, finders.AssetFinder)

        test_blotter = blotter.Blotter(test_finder)
        assert isinstance(test_blotter.portfolio, portfolio.Portfolio)
        assert isinstance(test_blotter.asset_finder, finders.AssetFinder)

        with pytest.raises(NotAPortfolioError):
            blotter.Blotter(portfolio='NOT A PORTFOLIO')

        with pytest.raises(NotAFinderError):
            blotter.Blotter(asset_finder='NOT A FINDER')



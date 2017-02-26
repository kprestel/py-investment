import pytest
import pytech.trading.blotter as blotter
import pytech.db.finders as finders
import pytech.fin.portfolio as portfolio
from pytech.utils.exceptions import NotAPortfolioError, NotAFinderError
from pytech.utils.enums import TradeAction, OrderSubType, OrderStatus, OrderType


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

    def test_cancel_order(self, populated_blotter):
        """
        Test canceling orders.

        :param populated_blotter:
        :type populated_blotter: blotter.Blotter
        """

        populated_blotter.cancel_order('one', 'AAPL')

        for k, v in populated_blotter:
                if k == 'one':
                    assert v.status == OrderStatus.CANCELLED.name

    def test_cancel_all_orders_for_asset(self, populated_blotter):
        """
        Test canceling all orders.

        :param populated_blotter:
        :type populated_blotter: blotter.Blotter
        """

        populated_blotter.cancel_all_orders_for_asset('AAPL')

        for order in populated_blotter:
            assert order.status == OrderStatus.CANCELLED.name






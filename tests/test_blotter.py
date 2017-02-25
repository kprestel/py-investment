import pytest
import pytech.trading.blotter as blotter
import pytech.db.finders as finders
import pytech.fin.portfolio as portfolio

class TestBlotter(object):

    def test_blotter_constructor(self):
        test_blotter = blotter.Blotter()

        assert isinstance(test_blotter.portfolio, portfolio.Portfolio)
        assert isinstance(test_blotter.asset_finder, finders.AssetFinder)


import pytest
import pytech.trading.blotter as b

from pytech.fin.asset import Fundamental, Stock


@pytest.fixture()
def blotter():
    return b.Blotter()


@pytest.fixture()
def populated_blotter(blotter):
    """Populate the blot and return it."""

    blotter.place_order('AAPL', 'BUY', 'LIMIT', 50, limit_price=100.10, order_id='one')
    blotter.place_order('AAPL', 'BUY', 'LIMIT', 50, limit_price=98.10, order_id='two')
    blotter.place_order('MSFT', 'SELL', 'LIMIT', 50, limit_price=93.10, order_id='three')
    blotter.place_order('FB', 'SELL', 'LIMIT', 50, limit_price=105.10, order_id='four')

    return blotter

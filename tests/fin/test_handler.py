from typing import TYPE_CHECKING

from backtest.event import SignalEvent
from fin.portfolio.handler import (
    SignalHandler,
    BasicSignalHandler,
)
from pytech.utils.enums import (
    TradeAction,
    SignalType,
    OrderType,
)

if TYPE_CHECKING:
    from fin.portfolio import Portfolio


class TestSignalHandler(object):
    pass

class TestBasicSignalHandler(object):

    def test_handle_long_signal(self, empty_portfolio: 'Portfolio'):
        event = SignalEvent('FB',
                            SignalType.LONG,
                            TradeAction.BUY,
                            limit_price=124.12)
        sig_handler = BasicSignalHandler()
        sig_handler._handle_long_signal(empty_portfolio, [], event)
        order_dict = empty_portfolio.blotter['FB']
        assert len(order_dict) == 1
        for id_, order in empty_portfolio.blotter:
            assert order.ticker == 'FB'
            assert order.order_type is OrderType.LIMIT
            assert order.id == id_
            assert order.limit_price == 124.12




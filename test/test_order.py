# noinspection PyUnresolvedReferences
import pytest

from pytech.backtest.event import SignalEvent
from pytech.trading.order import (
    Order,
    MarketOrder,
    LimitOrder,
)
from pytech.utils.enums import (
    TradeAction,
    OrderStatus,
    OrderType,
    SignalType,
)
import datetime as dt

import pytech.utils as utils

class TestOrder(object):
    """Tests for the base order class"""
    def test_get_order(self):
        ref = Order.get_order(OrderType.MARKET)
        mkt_order = ref('AAPL',
                        TradeAction.BUY,
                        100,
                        created=dt.datetime.now(),
                        order_id='1')
        assert isinstance(mkt_order, MarketOrder)

    def test_from_signal_event(self):
        event = SignalEvent('AAPL',
                            SignalType.LONG,
                            TradeAction.BUY,
                            limit_price=124.12)
        order = Order.from_signal_event(event, 100)
        assert isinstance(order, LimitOrder)
        assert order.qty == 100
        assert order.limit_price == 124.12


class TestMarketOrder(object):
    """Tests for the order class."""

    def test_constructor(self):
        created = dt.datetime(2017, 9, 1)
        market_buy_order = MarketOrder('AAPL',
                                       TradeAction.BUY,
                                       100,
                                       created=created,
                                       order_id='1')
        assert market_buy_order.triggered
        assert market_buy_order.qty == 100
        assert market_buy_order.id == '1'
        assert market_buy_order.status is OrderStatus.OPEN
        assert market_buy_order.created == utils.parse_date(created)
        assert market_buy_order.order_type is OrderType.MARKET
        assert market_buy_order.check_triggers(123, dt.datetime.now())

        market_sell_order = MarketOrder('AAPL',
                                       TradeAction.SELL,
                                       100,
                                       created=created,
                                       order_id='1')
        assert market_sell_order.qty == -100


class TestLimitOrder(object):
    """Test limit orders"""

    def test_constructor(self):
        created = dt.datetime(2017, 9, 1)

        limit_buy_order = LimitOrder('AAPL',
                                     TradeAction.BUY,
                                     100,
                                     created=created,
                                     order_id='1',
                                     limit_price=122.12222)
        assert not limit_buy_order.triggered
        assert limit_buy_order.qty == 100
        assert limit_buy_order.id == '1'
        assert limit_buy_order.order_type is OrderType.LIMIT
        assert limit_buy_order.limit_price == 122.12





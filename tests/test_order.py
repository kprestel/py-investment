# noinspection PyUnresolvedReferences
import pytest
import pandas as pd
import pytz

from pytech.trading.order import (
    Order,
    MarketOrder,
    LimitOrder,
)
from pytech.utils.enums import (
    TradeAction,
    OrderStatus,
    OrderType,
)
import datetime as dt


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
        assert market_buy_order.created == (pd.Timestamp(created)
                                            .replace(tzinfo=pytz.UTC))
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





import pytech.trading.order as ord
from pytech.utils.enums import OrderStatus, OrderType, TradeAction


class TestBlotter(object):
    def test_place_order(self, blotter):

        blotter.place_order(
                'AAPL', 'BUY', 'LIMIT', 50, limit_price=100.10, order_id='one')
        blotter.place_order(
                'MSFT', 'SELL', 'LIMIT', 50, limit_price=93.10,
                order_id='three')
        blotter.place_order(
                'FB', 'SELL', 'LIMIT', 50, limit_price=105.10,
                order_id='four')

        for k, v in blotter:
            assert isinstance(v, ord.Order)

    def test_cancel_order(self, populated_blotter):
        """
        Test canceling orders.

        :param populated_blotter:
        :type populated_blotter: blotter.Blotter
        """

        populated_blotter.cancel_order('one', 'AAPL')

        for k, v in populated_blotter:
            if k == 'one':
                assert v.status is OrderStatus.CANCELLED

    def test_cancel_all_orders_for_asset(self, populated_blotter):
        """
        Test canceling all orders.

        :param populated_blotter:
        :type populated_blotter: blotter.Blotter
        """

        populated_blotter.cancel_all_orders_for_asset('AAPL')

        for k, order in populated_blotter:
            if order.ticker == 'AAPL':
                assert order.status is OrderStatus.CANCELLED

    def test_create_order(self, blotter):
        stop_order = blotter._create_order('AAPL', TradeAction.BUY,
                                           50, OrderType.STOP,
                                           stop_price=110.1)
        assert stop_order.qty == 50
        assert isinstance(stop_order, ord.StopOrder)

        limit_order = blotter._create_order('AAPL', TradeAction.SELL,
                                            55, OrderType.LIMIT,
                                            limit_price=124.33)
        assert limit_order.limit_price == 124.33
        assert isinstance(limit_order, ord.LimitOrder)

        stop_limit_order = blotter._create_order('AAPL', TradeAction.SELL,
                                                 55, OrderType.STOP_LIMIT,
                                                 stop_price=111.2,
                                                 limit_price=124.33)
        assert stop_limit_order.limit_price == 124.33
        assert stop_limit_order.stop_price == 111.2
        assert isinstance(stop_limit_order, ord.StopLimitOrder)

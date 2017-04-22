import collections
import logging
import queue
from datetime import datetime
from typing import Dict

import pytech.utils.pandas_utils as pd_utils
from pytech.backtest.event import TradeEvent
from pytech.data.handler import DataHandler
from pytech.db.finders import AssetFinder
from pytech.fin.asset import Asset
from pytech.trading.commission import (AbstractCommissionModel,
                                       PerOrderCommissionModel)
from pytech.trading.order import (LimitOrder, MarketOrder, Order,
                                  StopLimitOrder, StopOrder, get_order_types)
from pytech.trading.trade import Trade
from pytech.utils.enums import (OrderStatus, OrderSubType, OrderType,
                                TradeAction)
from pytech.utils.exceptions import NotAFinderError

AnyOrder = get_order_types()


class Blotter(object):
    """Holds and interacts with all orders."""

    LOGGER_NAME = 'blotter'
    asset_finder: AssetFinder
    orders: Dict[str, AnyOrder]
    events: queue.Queue
    current_dt = datetime

    def __init__(self, events, asset_finder=None, commission_model=None):

        self.logger = logging.getLogger(self.LOGGER_NAME)
        self.asset_finder = asset_finder
        # dict of all orders. key=ticker of the asset, value=the asset.
        self.orders = {}
        # keep a record of all past trades.
        self.trades = []
        self.current_dt = None
        # events queue
        self.events = events
        self.bars = None

        if commission_model is None:
            self.commission_model = PerOrderCommissionModel()
        elif issubclass(commission_model.__class__, AbstractCommissionModel):
            self.commission_model = commission_model
        else:
            raise TypeError(
                    'commission_model must be a subclass of '
                    'AbstractCommissionModel. {} was provided'
                        .format(type(commission_model))
            )

    @property
    def bars(self):
        """Allow access to the :class:`DataHandler`"""
        return self._bars

    @bars.setter
    def bars(self, data_handler):
        if isinstance(data_handler, DataHandler):
            self._bars = data_handler
        elif data_handler is None:
            self._bars = None
        else:
            raise TypeError(
                    'bars must be an instance of DataHandler. {} was provided'
                        .format(type(data_handler))
            )

    @property
    def asset_finder(self):
        return self._asset_finder

    @asset_finder.setter
    def asset_finder(self, asset_finder):
        if asset_finder is not None and isinstance(asset_finder, AssetFinder):
            self._asset_finder = asset_finder
        elif asset_finder is None:
            self._asset_finder = AssetFinder()
        else:
            raise NotAFinderError(finder=type(asset_finder))

    def __getitem__(self, key):
        """Get an order from the orders dict."""
        return self.orders[key]

    def __setitem__(self, key, value):
        """
        Add an order to the orders dict.
        If the key is an instance of :class:`~ticker.Asset` then the ticker is used as the key, otherwise the key is the
        ticker.
        :param key: The key to dictionary, will always be the ticker of the ``Asset`` the order is for but an instance
        of :class:`~ticker.Asset` will also work as long as the ticker is set.
        :type key: Asset or str
        :param Order value: The order.
        """
        if issubclass(key.__class__, Asset):
            self.orders[key.ticker] = value
        else:
            self.orders[key] = value

    def __delitem__(self, key):
        """Delete an order from the orders dict."""
        del self.orders[key]

    def __iter__(self):
        """
        Iterate over the orders dict as well as the nested orders dict which 
        key=order_id and value=``Order``
        
        This means you can iterate over a :class:``Blotter`` instance directly 
        and access all of the open orders it has.
        """

        def do_iter(orders_dict):
            for k, v in orders_dict.items():
                if isinstance(v, collections.Mapping):
                    yield from do_iter(v)
                else:
                    yield k, v

        return do_iter(self.orders)

    def place_order(self, ticker: str, action: TradeAction,
                    order_type: OrderType, qty: int, stop_price: float = None,
                    limit_price: float = None, date_placed: datetime = None,
                    order_subtype: OrderSubType = None, order_id: str = None,
                    max_days_open: int = 90):
        """
        Open a new order.  If an open order for the given ``ticker`` already 
        exists placing a new order will **NOT** change the existing order, 
        it will be added to the tuple.

        :param ticker: The ticker of the :py:class:`~ticker.Asset` to place an 
        order for or the ticker of an ticker.
        :type ticker: Asset or str
        :param TradeAction or str action: **BUY** or **SELL**
        :param OrderType order_type: the type of order
        :param float stop_price: If creating a stop order this is the 
        stop price that will trigger the ``order``.
        :param float limit_price: If creating a limit order this is the price 
        that will trigger the ``order``.
        :param int qty: The number of shares to place an ``order`` for.
        :param datetime date_placed: The date and time the order is created.
        :param OrderSubType order_subtype: (optional) The type of order subtype
            (default: ``OrderSubType.DAY``)
        :param int max_days_open: Number of days to leave the ``order`` open 
        before it expires.
        :param str order_id: (optional) 
        The ID of the :class:`pytech.trading.order.Order`.
        :return: None
        """
        if qty == 0:
            # No point in making an order for 0 shares.
            return None

        if date_placed is None:
            date_placed = self.current_dt

        order = Order(
                ticker=ticker,
                action=action,
                order_type=order_type,
                order_subtype=order_subtype,
                stop=stop_price,
                limit=limit_price,
                qty=qty,
                created=date_placed,
                order_id=order_id,
                max_days_open=max_days_open
        )

        if ticker in self.orders:
            self.orders[ticker].update({
                order.id: order
            })
        else:
            self.orders[ticker] = {
                order.id: order
            }

    def _create_order(self,
                      ticker: str,
                      action: TradeAction,
                      qty: int,
                      order_type: OrderType = None,
                      *args,
                      **kwargs) -> AnyOrder:
        """
        Figure out what type of order to create based on given parameters.
        
        This is meant to be somewhat of an order factory.
        """
        if order_type is not None:
            return self._make_order(ticker, action, qty, order_type, **kwargs)

    def _make_order(self,
                    ticker: str,
                    action: TradeAction,
                    qty: int,
                    order_type: OrderType,
                    *args, **kwargs) -> AnyOrder:
        if order_type is OrderType.MARKET:
            return MarketOrder(ticker, action, qty)

        if order_type is OrderType.STOP:
            # TODO: should this handle error checking? or just let it raise.
            # stop_price = kwargs.pop('stop_price')
            return StopOrder(ticker, action, qty, **kwargs)

        if order_type is OrderType.LIMIT:
            # limit_price = kwargs.pop('limit_price')
            return LimitOrder(ticker, action, qty, **kwargs)

        if order_type is OrderType.STOP_LIMIT:
            # stop_price = kwargs.pop('stop_price')
            # limit_price = kwargs.pop('limit_price')
            return StopLimitOrder(
                    ticker, action, qty, **kwargs)

    def _find_order(self, order_id, ticker):
        if ticker is None:
            for asset_orders in self.orders:
                if order_id in asset_orders:
                    return asset_orders[order_id]
        else:
            for k, v in self.orders[ticker].items():
                if k == order_id:
                    return v

    def cancel_order(self, order_id, ticker=None, reason=''):
        """
        Mark an order as canceled so that it will not get executed.

        :param str order_id: The id of the order to cancel.
        :param ticker: (optional) The ticker that the order is associated with.
            Although it is not required to provide a ticker, 
            it is **strongly** encouraged.
            By providing a ticker the execution time of this method will 
            increase greatly.
        :param str reason: (optional) 
            The reason that the order is being cancelled.
        :return:
        """
        self._do_order_cancel(self._find_order(order_id, ticker), reason)

    def cancel_all_orders_for_asset(self, ticker, reason='', upper_price=None,
                                    lower_price=None, order_type=None,
                                    trade_action=None):
        """
        Cancel all orders for a given ticker's ticker and then clean up the 
        orders dict.

        :param str ticker: The ticker of the ticker to cancel all orders for.
        :param str reason: (optional) The reason for canceling the order.
        :param float lower_price: (optional) Only cancel orders 
        lower than this price.
        :param float upper_price: (optional) Only cancel orders greater than
        this price.
        :param OrderType order_type: (optional) Only cancel orders of the given
        order type.
        :param TradeAction trade_action: (optional) Only cancel orders that are
        either BUY or SELL.
        """
        for order in self.orders[ticker].values():
            self._do_order_cancel(order, reason)

    def _filter_orders(self, ticker, upper_price, lower_price, order_type,
                       trade_action):
        """
        Return an iterable of orders that meet the provided criteria.
        
        :param ticker: 
        :param upper_price: 
        :param lower_price: 
        :param OrderType order_type:
        :param TradeAction trade_action:
        :return: 
        """
        # TODO: this... how to filter efficiently?
        for order in self.orders[ticker].values():
            if order_type is not None and order.order_type is not order_type:
                continue
            elif trade_action is not None and order.action is not trade_action:
                continue

    def _do_order_cancel(self, order, reason):
        """
        Cancel any order that is passed to this method and log the appropriate 
        message.
        """
        if order.filled > 0:
            self.logger.warning(
                    'Order for ticker: {ticker} has been '
                    'partially filled. {amt} shares had already '
                    'been purchased.'
                        .format(ticker=order.ticker, amt=order.filled))
        elif order.filled < 0:
            self.logger.warning(
                    'Order for ticker: {ticker} has been partially filled. '
                    '{amt} shares had already been sold.'
                        .format(ticker=order.ticker, amt=order.filled))
        else:
            self.logger.info(
                    'Canceled order for ticker: {ticker} '
                    'successfully before it was executed.'
                        .format(ticker=order.ticker))
        order.cancel(reason)
        order.last_updated = self.current_dt

    def hold_order(self, order):
        """
        Place an order on hold. 
        
        :param order: 
        """
        self.orders[order.ticker][order.id].status = OrderStatus.HELD

    def hold_all_orders_for_asset(self, ticker):
        """
        Place all open orders for the given asset on hold.
        
        :param str ticker: The ticker of the asset to place all orders on hold.
        """
        for order in self.orders[ticker].values():
            self.hold_order(order)

    def reject_order(self, order_id, ticker=None, reason=''):
        """
        Mark an order as rejected. A rejected order is functionally the same as 
        a canceled order but an order being marked rejected is typically 
        involuntary or unexpected and comes from the broker. 
        Another case that an order will be rejected is if when the order 
        is being executed the owner does not have enough cash to fully execute it.

        :param str order_id: The id of the order being rejected.
        :param str ticker: (optional) 
            The ticker associated with the order being rejected.
        :param str reason: (optional) The reason the order was rejected.
        :return:
        """

        self._find_order(order_id, ticker).reject(reason)

        self.logger.warning(
                'Order id: {id} for ticker: {ticker} '
                'was rejected because: {reason}'
                    .format(id=order_id, ticker=ticker,
                            reason=reason or 'Unknown'))

    def check_order_triggers(self):
        """
        Check if any order has been triggered and if they have execute the 
        trade and then clean up closed orders.
        """
        for order_id, order in self:
            # should this be looking the close column?
            bar = self.bars.get_latest_bar(order.ticker)
            dt = bar[pd_utils.DATE_COL]
            current_price = bar[pd_utils.ADJ_CLOSE_COL]
            # available_volume = bar[pd_utils.VOL_COL]
            # check_triggers returns a boolean indicating if it is triggered.
            if order.check_triggers(dt=dt, current_price=current_price):
                self.events.put(
                        TradeEvent(order_id, current_price, order.qty, dt)
                )

    def make_trade(self, order, price_per_share, trade_date, volume):
        """
        Buy or sell an ticker from the ticker universe.

        :param str ticker: The ticker of the :class:``Asset`` to trade.
        :param TradeAction or str action: :py:class:``enum.TradeAction`` 
            see comments below.
        :param float price_per_share: the cost per share in the trade
        :param datetime trade_date: The date and time that the trade is 
            taking place.
        :return: ``order`` if the order is no longer open so it can be removed 
            from the ``portfolio`` order dict 
            and ``None`` if the order is still open
        :rtype: Order or None

        This method will add the ticker to the :py:class:``Portfolio`` ticker 
        dict and update the db to reflect the trade.

        Valid **action** parameter values are:

        * TradeAction.BUY
        * TradeAction.SELL
        * BUY
        * SELL
        """
        commission_cost = self.commission_model.calculate(order,
                                                          price_per_share)
        available_volume = order.get_available_volume(volume)
        avg_price_per_share = (
            ((price_per_share * available_volume) + commission_cost)
            / available_volume)

        order.commission += commission_cost

        trade = Trade.from_order(order, trade_date, commission_cost,
                                 price_per_share, available_volume,
                                 avg_price_per_share)

        order.filled += trade.qty
        self.trades.append(trade)
        return trade

    def purge_orders(self):
        """Remove any order that is no longer open."""
        open_orders = {}

        for ticker, asset_orders in self.orders:
            for order in asset_orders:
                if order.open and order.open_amount != 0:
                    open_orders[ticker] = order

        self.orders = open_orders

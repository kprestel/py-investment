import collections
import logging
import operator
import queue
from datetime import datetime
from typing import (
    List,
    TYPE_CHECKING,
    Union,
    Dict,
)

import pytech.utils as utils
from pytech.backtest.event import TradeEvent
from pytech.data.handler import DataHandler
from pytech.fin.asset.asset import Asset
from pytech.trading.commission import (
    AbstractCommissionModel,
    PerOrderCommissionModel,
)
from pytech.trading.order import (
    LimitOrder,
    MarketOrder,
    Order,
    StopLimitOrder,
    StopOrder,
)
from pytech.trading.trade import Trade
from pytech.utils.enums import (
    OrderStatus,
    OrderSubType,
    OrderType,
    TradeAction,
)

if TYPE_CHECKING:
    from fin.portfolio import Portfolio
    from . import (
        AnyOrder,
        TradingControl,
    )


class Blotter(object):
    """Holds and interacts with all orders."""

    def __init__(self,
                 events,
                 commission_model=None,
                 max_shares=None,
                 limit_pct_buffer: float = 1.02,
                 stop_pct_buffer: float = .98,
                 controls: List['TradingControl'] = None,
                 bars: 'DataHandler' = None) -> None:
        self.logger = logging.getLogger(__name__)
        # dict of all orders. key=ticker of the asset, value=the order.
        self.orders: Dict[str, Dict[str, 'AnyOrder']] = {}
        # keep a record of all past trades.
        self.trades: List[Trade] = []
        self.current_dt: datetime = None
        # events queue
        self.events: queue.Queue = events
        self.bars: 'DataHandler' = bars
        self.max_shares: int = max_shares or int(1e+11)
        # how much an auto generated limit price will be over the market price.
        self.limit_pct_buffer: float = limit_pct_buffer or 1.02
        self.stop_pct_buffer: float = stop_pct_buffer or .98
        if controls is None:
            self.controls = []
        else:
            self.controls = controls

        if commission_model is None:
            self.commission_model = PerOrderCommissionModel()
        elif issubclass(commission_model.__class__, AbstractCommissionModel):
            self.commission_model = commission_model
        else:
            raise TypeError(
                f'commission_model must be a subclass of '
                f'AbstractCommissionModel. '
                f'{type(commission_model)} was provided'
            )

    @property
    def bars(self) -> DataHandler:
        """Allow access to the :class:`DataHandler`"""
        return self._bars

    @bars.setter
    def bars(self, data_handler) -> None:
        if isinstance(data_handler, DataHandler):
            self._bars = data_handler
        elif data_handler is None:
            self._bars = None
        else:
            raise TypeError(f'bars must be an instance of DataHandler. '
                            f'{type(data_handler)} was provided')

    @property
    def current_dt(self):
        return self._current_dt

    @current_dt.setter
    def current_dt(self, val):
        if val is None:
            self._current_dt = None
        else:
            self._current_dt = utils.parse_date(val)

    def __getitem__(self, key) -> Union[Dict[str, 'AnyOrder'],
                                        'AnyOrder']:
        """
        Get an order from the orders dict or get all orders for a ticker.

        :param key: either an 'order_id` or a ticker.
            If `key` is an `order_id` then that order will be returned.
            If `key` is a ticker than a `Dict` of all the orders for the ticker
            will be returned
        """
        try:
            return self.orders[key]
        except KeyError:
            return self._find_order(key)

    def __setitem__(self, key, value):
        """
        Add an order to the orders dict.
        If the key is an instance of :class:`~ticker.Asset` then the ticker is used as the key, otherwise the key is the
        ticker.
        :param key: The key to dictionary, will always be the ticker of the
        ``Asset`` the order is for but an instance of :class:`~ticker.Asset`
        will also work as long as the ticker is set.
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

    def place_order(self,
                    portfolio: 'Portfolio',
                    ticker: str,
                    qty: int,
                    action: Union[TradeAction, str] = None,
                    order_type: Union[OrderType, str] = None,
                    stop_price: float = None,
                    limit_price: float = None,
                    date_placed: datetime = None,
                    order_subtype: OrderSubType = None,
                    order_id: str = None,
                    max_days_open: int = 90,
                    **kwargs):
        """
        Open a new order.  If an open order for the given ``ticker`` already
        exists placing a new order will **NOT** change the existing order,
        it will be added to the tuple.

        :param ticker: The ticker of the :py:class:`~ticker.Asset` to place an
        order for or the ticker of an ticker.
        :type ticker: Asset or str
        :param TradeAction or str action: **BUY** or **SELL**. If this is not
        provided it will be determined based on if ``qty`` is > 0.
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

        if action is None and qty < 0:
            action = TradeAction.SELL
        elif action is None and qty > 0:
            action = TradeAction.BUY

        if order_type is None:
            if action is TradeAction.SELL:
                order_type = OrderType.STOP
            elif action is TradeAction.BUY:
                order_type = OrderType.LIMIT
        else:
            OrderType.check_if_valid(order_type)

        is_limit_order = order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]
        is_stop_order = order_type in [OrderType.STOP, OrderType.STOP_LIMIT]

        curr_price = self.bars.latest_bar_value(ticker, utils.ADJ_CLOSE_COL)

        # set default stop/limit prices
        if is_limit_order and limit_price is None:
            limit_price = curr_price * self.limit_pct_buffer
        elif is_stop_order and stop_price is None:
            stop_price = curr_price * self.stop_pct_buffer

        if date_placed is None:
            date_placed = self.current_dt

        order = self._create_order(ticker,
                                   action,
                                   qty,
                                   order_type,
                                   stop_price=stop_price,
                                   limit_price=limit_price,
                                   date_placed=date_placed,
                                   order_subtype=order_subtype,
                                   order_id=order_id,
                                   max_days_open=max_days_open,
                                   **kwargs)

        for control in self.controls:
            control.validate(order, portfolio, self.current_dt, curr_price)

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
                      order_type: Union[OrderType, str],
                      **kwargs) -> Union[MarketOrder,
                                         StopOrder,
                                         LimitOrder,
                                         StopLimitOrder]:
        """
        Figure out what type of order to create based on given parameters.

        This is meant to be somewhat of an order factory.
        """
        order_type = OrderType.check_if_valid(order_type)

        if order_type is OrderType.MARKET:
            return MarketOrder(ticker, action, qty)

        if order_type is OrderType.STOP:
            return StopOrder(ticker, action, qty, **kwargs)

        if order_type is OrderType.LIMIT:
            return LimitOrder(ticker, action, qty, **kwargs)

        if order_type is OrderType.STOP_LIMIT:
            return StopLimitOrder(ticker, action, qty, **kwargs)

    def _find_order(self, order_id: str, ticker: str = None) -> 'AnyOrder':
        if ticker is None:
            for id_, order in self:
                if order_id == id_:
                    return order
        else:
            for k, v in self.orders[ticker].items():
                if k == order_id:
                    return v

    def cancel_order(self, order_id: str,
                     ticker: str = None,
                     reason: str = ''):
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
        order = self._find_order(order_id, ticker)
        self._do_order_cancel(order, reason)

    def cancel_all_orders_for_asset(self, ticker,
                                    reason='',
                                    upper_price=None,
                                    lower_price=None,
                                    order_type=None,
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
        either ``BUY`` or ``SELL``.
        """
        for order in self.orders[ticker].values():
            if self._check_filters(order, upper_price, lower_price,
                                   order_type, trade_action):
                self._do_order_cancel(order, reason)

    def _check_filters(self,
                       order: 'AnyOrder',
                       upper_price: float,
                       lower_price: float,
                       order_type: OrderType,
                       trade_action: TradeAction) -> bool:
        """Check all of the possible filters."""
        if self._filter_on_order_type(order, order_type):
            return True
        elif self._filter_on_trade_action(order, trade_action):
            return True
        elif self._filter_on_price(order,
                                   upper_price=upper_price,
                                   lower_price=lower_price):
            return True
        else:
            return False

    def _filter_on_order_type(self,
                              order: 'AnyOrder',
                              order_type: OrderType) -> bool:
        """
        Filter based on :class:``OrderType``. If the ``order`` is not the same
        :class:``OrderType`` as given then it will be filtered out from
        whatever action is being taken on the orders.

        :param order: The order to check whether it should be filtered or not.
        :param order_type: The type of order that the action should be taken
        on.
        :return: True if the order meets the criteria and the action should be
        taken.
        """
        if order_type is None:
            self.logger.debug('Order type was None. Filtering...')
            return True
        else:
            return order.order_type is order_type

    def _filter_on_trade_action(self,
                                order: 'AnyOrder',
                                trade_action: TradeAction):
        """

        :param order:
        :param trade_action:
        :return:
        """
        if trade_action is None:
            self.logger.debug('Trade action was none. Filtering...')
            return False
        else:
            return order.action is trade_action

    def _filter_on_price(self,
                         order: 'AnyOrder',
                         upper_price: float,
                         lower_price: float) -> bool:
        """
        Filter based on an upper and lower price and return ``True`` if the
        order meets the criteria. Meaning that whatever action is being
        taken on orders that match the given criteria should be taken on
        the given ``order``.

        If neither ``upper_price`` or ``lower_price`` are given then it is
        assumed that orders should be filtered based on price and
        ``False`` will be returned.

        :param order: The order that is being checked if it should be filtered
        or not.
        :param upper_price: The price that sets the upper limit for the
        filtering. Any order with a ``stop_price`` or ``limit_price`` above
        this amount **WILL** be filtered out.
        :param lower_price: Same as ``upper_price`` but any order with a
        ``stop_price`` or ``limit_price`` below this amount will be
        filtered out.
        :return: True if the order meets the criteria and the action should be
        taken on it.
        """
        if upper_price is None and lower_price is None:
            self.logger.warning('upper and lower price were both None.')
            return False

        if lower_price is None:
            return self._do_price_filter(order, upper_price, operator.gt)

        if upper_price is None:
            return self._do_price_filter(order, lower_price, operator.lt)

        lower_price_broken = self._do_price_filter(order, lower_price,
                                                   operator.lt)

        upper_price_broken = self._do_price_filter(order, upper_price,
                                                   operator.gt)

        return lower_price_broken or upper_price_broken

    def _do_price_filter(self,
                         order: 'AnyOrder',
                         price: float,
                         operator: operator) -> bool:
        """
        Filter based on stop and limit price.

        :param order:
        :param price:
        :return:
        """
        try:
            stop_price = order.stop_price
        except AttributeError:
            self.logger.debug('Order is not a stop order.')
            stop_broken = False
        else:
            stop_broken = operator(stop_price, price)

        try:
            limit_price = order.limit_price
        except AttributeError:
            self.logger.debug('Order is not a limit order.')
            limit_broken = False
        else:
            limit_broken = operator(limit_price, price)

        return limit_broken or stop_broken

    def _do_order_cancel(self, order: 'AnyOrder', reason: str):
        """
        Cancel any order that is passed to this method and log the appropriate
        message.
        """
        if order.filled > 0:
            self.logger.warning(f'Order for ticker: {order.ticker} has been '
                                f'partially filled. {order.filled} shares '
                                f'had already been purchased.')
        elif order.filled < 0:
            self.logger.warning(f'Order for ticker: {order.ticker} has been '
                                f'partially filled. {order.filled} shares '
                                'had already been sold.')
        else:
            self.logger.info(f'Canceled order for ticker: {order.ticker} '
                             'successfully before it was executed.')
        order.cancel(reason)
        order.last_updated = self.current_dt

    def hold_order(self, order):
        """
        Place an order on hold.

        :param order:
        """
        self.orders[order.ticker][order.id].status = OrderStatus.HELD

    def hold_all_orders_for_asset(self, ticker: str,
                                  upper_price: float = None,
                                  lower_price: float = None,
                                  order_type: OrderType = None,
                                  trade_action: TradeAction = None):
        """
        Place a hold on all orders for a given ticker's ticker.

        :param ticker: The ticker of the ticker to cancel all orders for.
        :param lower_price: (optional) Only hold orders
        lower than this price.
        :param upper_price: (optional) Only hold orders greater than
        this price.
        :param order_type: (optional) Only hold orders of the given
        order type.
        :param trade_action: (optional) Only hold orders that are
        either ``BUY`` or ``SELL``.
        """
        for order in self.orders[ticker].values():
            if self._check_filters(order, upper_price, lower_price,
                                   order_type, trade_action):
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
            f'Order id: {order_id} for ticker: {ticker} '
            f'was rejected because: {reason}')

    def check_order_triggers(self) -> List['AnyOrder']:
        """
        Check if any order has been triggered and if they have execute the
        trade and then clean up closed orders.
        """
        triggered_orders = []
        for order_id, order in self:
            # should this be looking the close column?
            bar = self.bars.get_latest_bar(order.ticker)
            dt = bar.name
            current_price = bar[utils.CLOSE_COL]
            # available_volume = bar[pd_utils.VOL_COL]
            # check_triggers returns a boolean indicating if it is triggered.
            if order.check_triggers(dt=dt, current_price=current_price):
                self.events.put(TradeEvent(order_id, current_price,
                                           order.qty, dt))
                triggered_orders.append(order)
        return triggered_orders

    def make_trade(self,
                   order: 'AnyOrder',
                   price_per_share: float,
                   trade_date: datetime,
                   volume: int) -> Trade:
        """
        Buy or sell an ticker from the ticker universe.

        :param volume:
        :param order:
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
        avg_price_per_share = _avg_price_per_share(price_per_share,
                                                   available_volume,
                                                   commission_cost)

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


def _avg_price_per_share(price_per_share: float, volume: float,
                         commission: float) -> float:
    """Calculate the average price per share."""
    total_cost = price_per_share * volume + commission
    return total_cost / volume

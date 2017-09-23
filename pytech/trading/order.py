import logging
import math
from abc import ABCMeta, abstractmethod
from datetime import datetime
from sys import float_info
from typing import (
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.offsets import DateOffset

import pytech.utils.common_utils as utils
import pytech.utils.dt_utils as dt_utils
from pytech.backtest.event import SignalEvent
from pytech.fin.asset.asset import Asset
from pytech.utils.enums import (OrderStatus, OrderSubType, OrderType,
                                TradeAction)
from pytech.utils.exceptions import BadOrderParams

logger = logging.getLogger(__name__)


class Order(metaclass=ABCMeta):
    """Hold open orders"""

    LOGGER_NAME = 'order'

    def __init__(self,
                 ticker: str,
                 action: TradeAction,
                 qty: int,
                 order_subtype: OrderSubType = None,
                 created: datetime = None,
                 max_days_open: int = None,
                 order_id: str = None,
                 *args, **kwargs):
        """
        Order constructor

        :param ticker: The ticker for which the order is associated with.
            This can either be an instance of an
            :class:`pytech.fin.ticker.Asset` or a string with of ticker
            of the ticker. If an ticker is passed in the ticker
            will be taken from it.
        :type ticker: Asset or str
        :param TradeAction action: Either BUY or SELL
        :param OrderSubType order_subtype: The order subtype to create
        default: :py:class:`pytech.enums.OrderSubType.DAY`
        :param int qty: The amount of shares the order is for.
        This should be negative if it is a sell order and positive if it is
        a buy order.
        :param datetime created: The date and time that the order was created
        :param int max_days_open: The max calendar days that an order can stay
            open without being cancelled.
            This parameter is not relevant to Day orders since they will be
            closed at the end of the day regardless.
            (default: None if the order_type is Day)
            (default: 90 if the order_type is not Day)
        :param str order_id: A uuid hex
        :raises NotAnAssetError: If the ticker passed in is not an ticker
        :raises InvalidActionError: If the action passed in is not a valid action
        :raises NotAPortfolioError: If the portfolio passed in is not a portfolio

        NOTES
        -----
        See :class:`pytech.enums.OrderType` to see valid order types
        See :class:`pytech.enums.OrderSubType` to see valid order sub types
        See :py:func:`asymmetric_round_price_to_penny` for more information on how
            `stop_price` and `limit_price` will get rounded.
        """
        super().__init__()
        self.id = order_id or utils.make_id()
        self.ticker = ticker
        self.logger = logging.getLogger(
                '{}_ticker_{}'.format(self.__class__.__name__, self.ticker))

        # TODO: validate that all of these inputs make sense together.
        # e.g. if its a stop order stop shouldn't be none
        self.action = TradeAction.check_if_valid(action)

        if order_subtype is not None:
            self.order_subtype = OrderSubType.check_if_valid(order_subtype)
        else:
            self.order_subtype = OrderSubType.DAY

        if self.order_subtype is OrderSubType.DAY:
            self.max_days_open = 1
        elif max_days_open is None:
            self.max_days_open = 90
        else:
            self.max_days_open = math.floor(max_days_open)

        self.qty = qty
        # How much commission has already been charged on the order.
        self.commission = 0.0
        self.filled = 0
        self._status = OrderStatus.OPEN
        self.reason = None

        if created is not None:
            self.created = dt_utils.parse_date(created)
        else:
            self.created = pd.Timestamp(datetime.now())

        # the last time the order changed
        self.last_updated = self.created
        self.close_date = None

    @property
    def status(self):
        if not self.open_amount:
            return OrderStatus.FILLED
        elif self._status == OrderStatus.HELD and self.filled:
            return OrderStatus.OPEN
        else:
            return self._status

    @status.setter
    def status(self, status):
        self._status = OrderStatus.check_if_valid(status)

    @property
    def ticker(self):
        """Make ticker always return the ticker unless directly accessed."""
        if issubclass(self._ticker.__class__, Asset):
            return self._ticker.ticker
        else:
            return self._ticker

    @ticker.setter
    def ticker(self, ticker):
        """
        If an ticker is passed in then use it otherwise use the
        string passed in.
        """

        self._ticker = ticker

    @property
    def qty(self):
        return self._qty

    @qty.setter
    def qty(self, qty):
        """
        Ensure qty is an integer and if it is a **sell** order
        qty should be negative.
        """

        if self.action is TradeAction.SELL:
            if int(qty) > 0:
                # qty should be negative if it is a sell order.
                self._qty = int(qty * -1)
            else:
                self._qty = int(qty)
        else:
            self._qty = int(qty)

    @property
    @abstractmethod
    def triggered(self) -> bool:
        """
        For a market order, True.
        For a stop order, True IF stop_reached.
        For a limit order, True IF limit_reached.
        """

    @property
    @abstractmethod
    def order_type(self) -> OrderType:
        """Must return the OrderType"""

    @property
    def open(self):
        return self.status in [OrderStatus.OPEN, OrderStatus.HELD]

    @property
    def open_amount(self):
        return self.qty - self.filled

    def cancel(self, reason=''):
        self.status = OrderStatus.CANCELLED
        self.reason = reason

    def reject(self, reason=''):
        self.status = OrderStatus.REJECTED
        self.reason = reason

    def hold(self, reason=''):
        self.status = OrderStatus.HELD
        self.reason = reason

    @abstractmethod
    def check_triggers(self, current_price: float, dt: datetime) -> bool:
        """
        Check if any of the ``order``'s limits have been broken in a way that
        would trigger the ``order``.

        :param datetime dt: The current datetime.
        :param float current_price: The current price to check the triggers
            against.
        :return: True if the order is triggered otherwise False
        :rtype: bool
        """

    def check_order_expiration(self, current_date=datetime.now()):
        """
        Check if the order should be closed due to passage of time and update
        the order's status.

        :param datetime current_date: This is used to facilitate backtesting,
        so that the current date can be mocked in order to accurately
        trigger/cancel orders in the past.
        (default: datetime.now())
        """
        trading_cal = mcal.get_calendar(self.portfolio.trading_cal)
        schedule = trading_cal.schedule(start_date=self.portfolio.start_date,
                                        end_date=self.portfolio.end_date)

        if self.order_subtype is OrderSubType.DAY:
            if not trading_cal.open_at_time(schedule,
                                            pd.Timestamp(current_date)):
                reason = 'Market closed without executing order.'
                self.logger.info(
                        'Canceling trade for ticker: {} due to {}'.format(
                                self.ticker.ticker, reason))
                self.cancel(reason=reason)
        elif self.order_subtype is OrderSubType.GOOD_TIL_CANCELED:
            expr_date = self.created + DateOffset(days=self.max_days_open)
            # check if the expiration date is today.
            if current_date.date() == expr_date.date():
                # if the expiration date is today then
                # check if the market has closed.
                if not trading_cal.open_at_time(schedule,
                                                pd.Timestamp(current_date)):
                    reason = ('Max days of {} had passed without the '
                              'underlying order executing.'
                              .format(self.max_days_open))
                    self.logger.info(
                            'Canceling trade for ticker: {} due to {}'.format(
                                    self.ticker.ticker, reason))
                    self.cancel(reason=reason)
        else:
            return

    def get_available_volume(self, available_volume):
        """
        Get the available volume to trade.

        This will the min of open_amount and the assets volume.

        :param int available_volume: The amount of shares available to trade
            at a given point in time.
        :return: The number of shares available to trade
        :rtype: int
        """
        return int(min(available_volume, abs(self.open_amount)))

    @classmethod
    def from_signal_event(cls, signal: SignalEvent, action: TradeAction):
        """
        Create an order from a :class:`pytech.backtest.event.SignalEvent`.

        :param SignalEvent or LongSignalEvent or ShortSignalEvent signal:
        The signal event that triggered the order to be created.
        :param TradeAction action: The TradeAction that the order is for.
        :return: A new order
        :rtype: Order
        """
        return cls(signal.ticker, action, signal.order_type,
                   stop=signal.stop_price, limit=signal.limit_price)

    @classmethod
    def get_order(cls, type_: Union['Order', str]):
        """
        Get a reference to the Order class requested.

        :param type_: the type of order class to get.
        :return: the reference to the class
        """
        for subclass in cls.__subclasses__():
            if subclass.order_type is OrderType.check_if_valid(type_):
                return subclass


class MarketOrder(Order):
    """Orders that will be executed at whatever the latest market price is"""

    def __init__(self, ticker, action, qty, created=None, order_id=None,
                 *args, **kwargs):
        super().__init__(ticker, action, qty, created=created,
                         order_id=order_id, *args, **kwargs)

    def check_triggers(self, current_price: float, dt: datetime) -> bool:
        return True

    @property
    def triggered(self) -> bool:
        return True

    @property
    def order_type(self) -> OrderType:
        return OrderType.MARKET


class LimitOrder(Order):
    """Limit order. Update this."""

    def __init__(self,
                 ticker: str,
                 action: TradeAction,
                 qty: int,
                 order_subtype: OrderSubType = None,
                 created: datetime = None,
                 max_days_open: int = None,
                 order_id: str = None,
                 *,
                 limit_price: float,
                 **kwargs):
        super().__init__(ticker, action, qty, order_subtype, created,
                         max_days_open, order_id, **kwargs)
        self.limit_reached = False
        self.limit_price = limit_price

    @property
    def limit_price(self) -> float:
        return self._limit_price

    @limit_price.setter
    def limit_price(self, limit_price: float) -> None:
        """
        Convert from a float to a 2 decimal point number that rounds
        favorably based on the trade_action
        """
        pref_round_down = self.action is TradeAction.BUY

        try:
            if np.isfinite(limit_price):
                self._limit_price = asymmetric_round_price_to_penny(
                        limit_price, pref_round_down)
        except TypeError:
            raise BadOrderParams(order_type='limit', price=limit_price)

    @property
    def triggered(self) -> bool:
        return self.limit_reached

    @property
    def order_type(self) -> OrderType:
        return OrderType.LIMIT

    def check_triggers(self, current_price: float, dt: datetime) -> bool:
        """
        Check if the ``order``'s limit price has been broken.

        Update the state of the order if it has.

        :param current_price:
        :param dt:
        :return:
        """
        if self.action is TradeAction.BUY and current_price <= self.limit_price:
            self.limit_reached = True
            self.last_updated = dt
        elif current_price >= self.limit_price:
            # The only other actions are SELL and EXIT which are both sell.
            self.limit_reached = True
            self.last_updated = dt
        else:
            # Update the updated date to show the last time it was checked.
            self.last_updated = dt

        return self.triggered


class StopOrder(Order):
    """Stop orders."""

    def __init__(self,
                 ticker: str,
                 action: TradeAction,
                 qty: int,
                 order_subtype: OrderSubType = None,
                 created: datetime = None,
                 max_days_open: int = None,
                 order_id: str = None,
                 *,
                 stop_price: float,
                 **kwargs):
        super().__init__(ticker, action, qty, order_subtype, created,
                         max_days_open, order_id, **kwargs)
        self.stop_price = stop_price
        self.stop_reached = False

    @property
    def stop_price(self):
        return self._stop_price

    @stop_price.setter
    def stop_price(self, stop_price):
        """
        Convert from a float to a 2 decimal point number that rounds
        favorably based on the trade_action
        """
        pref_round_down = self.action is not TradeAction.BUY
        try:
            if np.isfinite(stop_price):
                self._stop_price = asymmetric_round_price_to_penny(
                        stop_price, pref_round_down)
        except TypeError:
            raise BadOrderParams(order_type='stop', price=stop_price)

    @property
    def triggered(self) -> bool:
        return self.stop_reached

    @property
    def order_type(self) -> OrderType:
        return OrderType.STOP

    def check_triggers(self, current_price: float, dt: datetime) -> bool:
        if self.action is TradeAction.BUY and current_price >= self.stop_price:
            self.stop_reached = True
            self.last_updated = dt
        elif current_price <= self.stop_price:
            self.stop_reached = True
            self.last_updated = dt
        else:
            self.last_updated = dt

        return self.triggered


class StopLimitOrder(StopOrder, LimitOrder):
    """Stop limit"""

    def __init__(self,
                 ticker: str,
                 action: TradeAction,
                 qty: int,
                 stop_price: float,
                 limit_price: float,
                 order_subtype: OrderSubType = None,
                 created: datetime = None,
                 max_days_open: int = None,
                 order_id: str = None,
                 **kwargs):
        # # stop_price = kwargs.pop('stop_price')
        # limit_price = kwargs.pop('limit_price')
        super().__init__(ticker, action, qty,
                         stop_price=stop_price,
                         limit_price=limit_price,
                         order_subtype=order_subtype, created=created,
                         max_days_open=max_days_open, order_id=order_id,
                         **kwargs)

    @property
    def triggered(self) -> bool:
        return self.stop_reached and self.limit_reached

    @property
    def order_type(self) -> OrderType:
        return OrderType.STOP_LIMIT

    def check_triggers(self, current_price: float, dt: datetime) -> bool:
        """
        Call both the :class:``StopOrder`` and the :class:``LimitOrder``
        :func:``check_triggers`` in order to check if both the stop trigger
        and the limit trigger have been met.

        :param current_price:
        :param dt:
        :return:
        """
        if not self.stop_reached:
            StopOrder.check_triggers(self, current_price, dt)
        if not self.limit_reached:
            LimitOrder.check_triggers(self, current_price, dt)

        return self.triggered


def asymmetric_round_price_to_penny(price, prefer_round_down,
                                    diff=(0.0095 - .005)):
    """
    This method was taken from the zipline lib.

    Asymmetric rounding function for adjusting prices to two places in a way
    that "improves" the price.  For limit prices, this means preferring to
    round down on buys and preferring to round up on sells.  For stop prices,
    it means the reverse.

    If prefer_round_down == True:
        When .05 below to .95 above a penny, use that penny.
    If prefer_round_down == False:
        When .95 below to .05 above a penny, use that penny.

    In math-speak:
    If prefer_round_down: [<X-1>.0095, X.0195) -> round to X.01.
    If not prefer_round_down: (<X-1>.0005, X.0105] -> round to X.01.
    """

    # Subtracting an epsilon from diff to enforce the open-ness of the upper
    # bound on buys and the lower bound on sells.  Using the actual system
    # epsilon doesn't quite get there, so use a slightly less epsilon-ey value.
    epsilon = float_info.epsilon * 10
    diff -= epsilon

    # relies on rounding half away from zero, unlike numpy's bankers' rounding
    rounded = round(price - (diff if prefer_round_down else -diff), 2)
    if np.isclose([rounded], [0.0]):
        return 0.0
    return rounded

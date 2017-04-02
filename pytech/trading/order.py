import logging
from datetime import datetime
from sys import float_info

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.offsets import DateOffset

import pytech.utils.common_utils as utils
import pytech.utils.dt_utils as dt_utils
from pytech.fin.asset import Asset
from pytech.utils.enums import OrderStatus, OrderSubType, OrderType, \
    TradeAction

logger = logging.getLogger(__name__)


class Order(object):
    """Hold open orders"""

    LOGGER_NAME = 'order'

    def __init__(self, ticker, action, order_type, order_subtype=None,
                 stop=None, limit=None, qty=0,
                 filled=0, created=None, max_days_open=None, id=None):
        """
        Order constructor

        :param ticker: The ticker for which the order is associated with.  
            This can either be an instance of an
            :class:`pytech.fin.ticker.Asset` or a string with of ticker 
            of the ticker. If an ticker is passed in the ticker
            will be taken from it.
        :type ticker: Asset or str
        :param Portfolio blot: The :py:class:`pytech.blot.Blotter` 
            that the ticker is associated with
        :param TradeAction action: Either BUY or SELL
        :param OrderType order_type: The type of order to create.
            Also can be a str.
        :param OrderSubType order_subtype: The order subtype to create
            default: :py:class:`pytech.enums.OrderSubType.DAY`
        :param float stop: The price at which to execute a stop order. 
            If this is not a stop order then leave as None.
        :param float limit: The price at which to execute a limit order. 
            If this is not a limit order then leave as None.
        :param int qty: The amount of shares the order is for.
            This should be negative if it is a sell order and positive if it is 
            a buy order.
        :param int filled: How many shares of the order have already been 
            filled, if any.
        :param float commission: The amount of commission that has already been 
            charged on the order.
        :param datetime created: The date and time that the order was created
        :param int max_days_open: The max calendar days that an order can stay 
            open without being cancelled.
            This parameter is not relevant to Day orders since they will be 
            closed at the end of the day regardless.
            (default: None if the order_type is Day)
            (default: 90 if the order_type is not Day)
        :param str id: A uuid hex
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
        self.id = id or utils.make_id()
        self.logger = logging.getLogger(
                '{}_id_{}'.format(self.LOGGER_NAME, self.id))

        # TODO: validate that all of these inputs make sense together.
        # e.g. if its a stop order stop shouldn't be none
        self.action = TradeAction.check_if_valid(action)
        self.order_type = OrderType.check_if_valid(order_type)

        if order_subtype is not None:
            self.order_subtype = OrderSubType.check_if_valid(order_subtype)
        else:
            self.order_subtype = OrderSubType.DAY

        self.ticker = ticker

        if self.order_subtype is OrderSubType.DAY:
            self.max_days_open = 1
        elif max_days_open is None:
            self.max_days_open = 90
        else:
            self.max_days_open = int(max_days_open)

        self.qty = qty
        # How much commission has already been charged on the order.
        self.commission = 0.0
        self.stop_price = stop
        self.limit_price = limit
        self.stop_reached = False
        self.limit_reached = False
        self.filled = filled
        self._status = OrderStatus.OPEN
        self.reason = None

        if created is not None:
            self.created = dt_utils.parse_date(created)
        else:
            self.created = pd.Timestamp(datetime.now())

        # the last time the order changed
        self.last_updated = self.created
        self.close_date = None

        if (self.stop_price is None
            and self.limit_price is None
            and self.order_type is not OrderType.MARKET):
            self.logger.warning(
                    'stop_price and limit_price were both None and OrderType '
                    'was not MARKET. Changing order_type to a MARKET order')
            self.order_type = OrderType.MARKET

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
    def stop_price(self):
        return self._stop_price

    @stop_price.setter
    def stop_price(self, stop_price):
        """
        Convert from a float to a 2 decimal point number that rounds 
        favorably based on the trade_action
        """
        if stop_price is not None:
            pref_round_down = self.action is not TradeAction.BUY
            self._stop_price = asymmetric_round_price_to_penny(stop_price,
                                                               pref_round_down)
        else:
            self._stop_price = None

    @property
    def limit_price(self):
        return self._limit_price

    @limit_price.setter
    def limit_price(self, limit_price):
        """
        Convert from a float to a 2 decimal point number that rounds 
        favorably based on the trade_action
        """
        if limit_price is not None:
            pref_round_down = self.action is TradeAction.BUY
            self._limit_price = asymmetric_round_price_to_penny(limit_price,
                                                                pref_round_down)
        else:
            self._limit_price = None

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
    def triggered(self):
        """
        For a market order, True.
        For a stop order, True IF stop_reached.
        For a limit order, True IF limit_reached.
        """
        if self.order_type is OrderType.MARKET:
            return True

        if self.stop_price is not None and not self.stop_reached:
            return False

        if self.limit_price is not None and not self.limit_reached:
            return False

        return True

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

    def check_triggers(self, current_price, dt):
        """
        Check if any of the ``order``'s limits have been broken in a way that 
        would trigger the ``order``.

        :param datetime dt: The current datetime.
        :param float current_price: The current price to check the triggers 
            against.
        :return: True if the order is triggered otherwise False
        :rtype: bool
        """

        if self.order_type is OrderType.MARKET or self.triggered:
            return True

        if self.order_type is OrderType.STOP_LIMIT and self.action is TradeAction.BUY:
            if current_price >= self.stop_price:
                self.stop_reached = True
                self.last_updated = dt
                if current_price >= self.limit_price:
                    self.limit_reached = True
        elif self.order_type is OrderType.STOP_LIMIT and self.action is TradeAction.SELL:
            if current_price <= self.stop_price:
                self.stop_reached = True
                self.last_updated = dt
                if current_price >= self.limit_price:
                    self.limit_reached = True
        elif self.order_type is OrderType.STOP and self.action is TradeAction.BUY:
            if current_price >= self.stop_price:
                self.stop_reached = True
                self.last_updated = dt
        elif self.order_type is OrderType.STOP and self.action is TradeAction.SELL:
            if current_price <= self.stop_price:
                self.stop_reached = True
                self.last_updated = dt
        elif self.order_type is OrderType.LIMIT and self.action is TradeAction.BUY:
            if current_price >= self.limit_price:
                self.limit_reached = True
                self.last_updated = dt
        elif self.order_type is OrderType.LIMIT and self.action is TradeAction.SELL:
            if current_price <= self.limit_price:
                self.limit_reached = True
                self.last_updated = dt

        if self.stop_reached and self.order_type is OrderType.STOP_LIMIT:
            # change the STOP_LIMIT order to a LIMIT order
            self.stop_price = None
            self.order_type = OrderType.LIMIT

        return self.triggered

    def check_order_expiration(self, current_date=datetime.now()):
        """
        Check if the order should be closed due to passage of time and update the order's status.

        :param datetime current_date: This is used to facilitate backtesting, so that the current date can be mocked in order to
            accurately trigger/cancel orders in the past.
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
                # if the expiration date is today then check if the market has closed.
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

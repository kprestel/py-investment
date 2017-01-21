from datetime import datetime
from sys import float_info
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import pandas_market_calendars as mcal
from dateutil.relativedelta import relativedelta

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Numeric, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy_utils import generic_relationship

from pytech import Base, utils
import pytech.db_utils as db
from pytech.enums import TradeAction, OrderStatus, OrderType, OrderSubType
from pytech.exceptions import NotAnAssetError, PyInvestmentError, InvalidActionError, NotAPortfolioError, \
    UntriggeredTradeError
from pytech.asset import Asset
import logging

logger = logging.getLogger(__name__)


class Order(Base):
    """Hold open orders"""

    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer)
    asset_type = Column(String)
    asset = generic_relationship(asset_id, asset_type)
    portfolio_id = Column(Integer, ForeignKey('portfolio.id'), primary_key=True)
    status = Column(String)
    created = Column(DateTime)
    close_date = Column(DateTime)
    commission = Column(Numeric)
    stop_price = Column(Numeric)
    limit_price = Column(Numeric)
    stop_reached = Column(Boolean)
    limit_reached = Column(Boolean)
    qty = Column(Integer)
    filled = Column(Integer)
    action = Column(String)
    reason = Column(String)
    order_type = Column(String)
    LOGGER_NAME = 'order'

    def __init__(self, asset, portfolio, action, order_type, order_subtype=None, stop=None, limit=None, qty=0,
                 filled=0, commission=0, created=datetime.now(), max_days_open=None):
        """
        Order constructor

        :param Asset asset: The asset for which the order is associated with
        :param Portfolio portfolio: The :py:class:`pytech.portfolio.Portfolio` that the asset is associated with
        :param TradeAction action: Either BUY or SELL
        :param OrderType order_type: The type of order to create.
            Also can be a str.
        :param OrderSubType order_subtype: The order subtype to create
            default: :py:class:`pytech.enums.OrderSubType.DAY`
        :param float stop: The price at which to execute a stop order. If this is not a stop order then leave as None
        :param float limit: The price at which to execute a limit order. If this is not a limit order then leave as None
        :param int qty: The amount of shares the order is for.
            This should be negative if it is a sell order and positive if it is a buy order.
        :param int filled: How many shares of the order have already been filled, if any.
        :param float commission: The amount of commission associated with placing the order.
        :param datetime created: The date and time that the order was created
        :param int max_days_open: The max calendar days that an order can stay open without being cancelled.
            This parameter is not relevant to Day orders since they will be closed at the end of the day regardless.
            default: None if the order_type is Day
            default: 90 if the order_type is not Day
        :raises NotAnAssetError: If the asset passed in is not an asset
        :raises InvalidActionError: If the action passed in is not a valid action
        :raises NotAPortfolioError: If the portfolio passed in is not a portfolio

        NOTES
        -----
        See :class:`pytech.enums.OrderType` to see valid order types
        See :class:`pytech.enums.OrderSubType` to see valid order sub types
        """
        from pytech import Portfolio

        self.logger = logging.getLogger(self.LOGGER_NAME)

        if issubclass(asset.__class__, Asset):
            self.asset = asset
        else:
            raise NotAnAssetError(asset=type(asset))

        if isinstance(portfolio, Portfolio):
            self.portfolio = portfolio
        else:
            raise NotAPortfolioError(portfolio=type(portfolio))

        # TODO: validate that all of these inputs make sense together. e.g. if its a stop order stop shouldn't be none
        self.action = TradeAction.check_if_valid(action)
        self.order_type = OrderType.check_if_valid(order_type)

        self.order_subtype = OrderSubType.check_if_valid(order_subtype) or OrderSubType.DAY

        if self.order_subtype is OrderSubType.DAY:
            self.max_days_open = 1
        elif max_days_open is None:
            self.max_days_open = 90
        else:
            self.max_days_open = int(max_days_open)

        self._qty = qty
        self.commission = commission
        self.stop_price = stop
        self.limit_price = limit
        self.stop_reached = False
        self.limit_reached = False
        self.filled = filled
        self._status = OrderStatus.OPEN
        self.reason = None
        self.created = utils.parse_date(created)
        # the last time the order changed
        self.last_updated = self.created
        self.close_date = None

        if self.stop_price is None and self.limit_price is None and self.order_type is not OrderType.MARKET:
            self.logger.warning('stop_price and limit_price were both None and OrderType was not MARKET. Changing '
                                'order_type to a MARKET order')
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
    def qty(self):
        return self._qty

    @qty.setter
    def qty(self, qty):
        """Ensure qty is an integer and if it is a **sell** order qty should be negative."""

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

    def check_triggers(self, current_price=None, dt=None):
        """
        Check if any of the order's triggers should be pulled and execute a trade and then delete the order.

        :param datetime dt: The current datetime.
            (default: ``datetime.now()``)
        :param float current_price: The current price to check the triggers against.
            (default: ``None``)
            If left at the default then the current price will retrieved.
        :return: True if the order is triggered otherwise False
        :rtype: bool
        """

        if self.order_type is OrderType.MARKET or self.triggered:
            return True

        if current_price is None and dt is None:
            current_price = self.asset.get_price_quote()
            current_price = current_price.price
            dt = datetime.now()
        elif current_price is None:
            current_price = self.asset.get_price_quote(d=dt)
            current_price = current_price.price

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
        schedule = trading_cal.schedule(start_date=self.portfolio.start_date, end_date=self.portfolio.end_date)

        if self.order_subtype is OrderSubType.DAY:
            if not trading_cal.open_at_time(schedule, pd.Timestamp(current_date)):
                reason = 'Market closed without executing order.'
                self.logger.info('Canceling trade for asset: {} due to {}'.format(self.asset.ticker, reason))
                self.cancel(reason=reason)
        elif self.order_subtype is OrderSubType.GOOD_TIL_CANCELED:
            expr_date = self.created + DateOffset(days=self.max_days_open)
            # check if the expiration date is today.
            if current_date.date() == expr_date.date():
                # if the expiration date is today then check if the market has closed.
                if not trading_cal.open_at_time(schedule, pd.Timestamp(current_date)):
                    reason = 'Max days of {} had passed without the underlying order executing.'.format(
                        self.max_days_open)
                    self.logger.info('Canceling trade for asset: {} due to {}'.format(self.asset.ticker, reason))
                    self.cancel(reason=reason)
        else:
            return


    def get_available_volume(self, dt):
        """
        Get the available volume to trade.  This will the min of open_amount and the assets volume.
        :param datetime dt:
        :return: The number of shares available to trade
        :rtype: int
        """

        return int(min(self.asset.get_volume(dt=dt), abs(self.open_amount)))



class Trade(Base):
    """
    This class is used to make trades and keep trade of past trades.

    Trades must be created as a result of an :class:``Order`` executing.
    """
    id = Column(Integer, primary_key=True)
    trade_date = Column(DateTime)
    action = Column(String)
    strategy = Column(String)
    qty = Column(Integer)
    price_per_share = Column(Numeric)
    corresponding_trade_id = Column(Integer, ForeignKey('trade.id'))
    net_trade_value = Column(Numeric)
    ticker = Column(String)
    order = relationship('Order', backref='trade')
    order_id = Column(Integer, ForeignKey('order.id'))
    commission = Column(Integer)

    # owned_stock_id = Column(Integer, ForeignKey('owned_stock.id'))
    # owned_stock = relationship('OwnedStock')
    # corresponding_trade = relationship('Trade', remote_side=[id])

    def __init__(self, qty, price_per_share, action, strategy, order, commission=0.0, trade_date=None, ticker=None):
        """
        :param datetime trade_date: corresponding to the date and time of the trade date
        :param int qty: number of shares traded
        :param float price_per_share: price per individual share in the trade or the average share price in the trade
        :param Asset ticker: a :py:class:`~.asset.Asset`, the ``asset`` object that was traded
        :param Order order: a :py:class:`~.order.Order` that was executed as a result of the order executing
        :param float commission: the amount of commission paid to execute this trade
            (default: 0.0)
        :param TradeAction or str action: :py:class:`~.enum.TradeAction`
        :param str position: must be *long* or *short*

        .. note::

        Valid **action** parameter values are:

        * TradeAction.BUY
        * TradeAction.SELL
        * BUY
        * SELL

        `commission` is not necessarily the same commission associated with the ``order`` this will depend on the
        type of :class:``AbstractCommissionModel`` used.
        """

        if trade_date:
            self.trade_date = utils.parse_date(trade_date)
        else:
            self.trade_date = datetime.now()

        self.action = TradeAction.check_if_valid(action)
        self.strategy = strategy
        self.ticker = ticker
        self.qty = qty
        self.price_per_share = price_per_share
        self.corresponding_trade_id = self._get_corresponding_trade_id(ticker=ticker)
        self.order = order
        self.order_id = order.id
        self.commission = commission

    @classmethod
    def _get_corresponding_trade_id(cls, ticker):
        """Get the most recent trade's id"""

        with db.query_session() as session:
            corresponding_trade = session.query(cls) \
                .filter(cls.ticker == ticker) \
                .order_by(cls.trade_date.desc()) \
                .first()
        try:
            return corresponding_trade.id
        except AttributeError:
            return None

    @classmethod
    def from_order(cls, order, trade_date, execution_price=None, strategy=None):
        """
        Make a trade from a triggered order object.

        :param Order order: The ``order`` object creating the trade.
        :param float execution_price: The price that the trade will be executed at.
        :raises UntriggeredTradeError: if the order has not been triggered.
        :return: a new ``trade``
        :rtype: Trade
        """

        if not order.triggered:
            raise UntriggeredTradeError(id=order.id)

        if execution_price is None:
            exec_price = order.asset.get_price_quote()
            execution_price = exec_price.price
        else:
            execution_price = execution_price

        if strategy is None:
            strategy = order.order_type.name

        trade_dict = {
            'ticker': order.asset.ticker,
            'qty': order.get_available_volume(dt=trade_date),
            'action': order.action,
            'price_per_share': execution_price,
            'strategy': strategy,
            'order': order,
            'trade_date': trade_date
        }

        return cls(**trade_dict)


def asymmetric_round_price_to_penny(price, prefer_round_down, diff=(0.0095 - .005)):
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
    diff = diff - epsilon

    # relies on rounding half away from zero, unlike numpy's bankers' rounding
    rounded = round(price - (diff if prefer_round_down else -diff), 2)
    if np.isclose([rounded], [0.0]):
        return 0.0
    return rounded

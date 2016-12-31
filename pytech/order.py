from datetime import datetime
from sys import float_info
import numpy as np

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Numeric, Boolean
from sqlalchemy_utils import generic_relationship

from pytech import Base, utils
import pytech.db_utils as db
from pytech.enums import TradeAction, OrderStatus, OrderType, OrderSubType
from pytech.exceptions import NotAnAssetError, PyInvestmentError, InvalidActionError
from pytech.stock import Asset



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
    stop = Column(Numeric)
    limit = Column(Numeric)
    stop_reached = Column(Boolean)
    limit_reached = Column(Boolean)
    qty = Column(Integer)
    filled = Column(Integer)
    action = Column(String)
    reason = Column(String)
    order_type = Column(String)

    def __init__(self, asset, portfolio, action, order_type, stop=None, limit=None, qty=0, filled=0, commission=0,
                 created=datetime.now(), order_subtype=OrderSubType.DAY):
        """
        Order constructor

        :param asset: The asset for which the order is associated with
        :type asset: Asset
        :param portfolio: The portfolio that the asset is associated with
        :type portfolio: Portfolio
        :param action: Either BUY or SELL
        :type action: TradeAction
        :param order_type: The type of order to create.
        :type order_type: str, or :class:``pytech.enums.OrderType``
        :param order_subtype: The order subtype to create
            default: :class:``pytech.enums.OrderSubType.DAY``
        :type order_subtype: str or :class:``pytech.enums.OrderSubType``
        :param stop: The price at which to execute a stop order. If this is not a stop order then leave as None
        :type stop: str
        :param limit: The price at which to execute a limit order. If this is not a limit order then leave as None
        :type limit: str
        :param qty: The amount of shares the order is for.
            This should be negative if it is a sell order and positive if it is a buy order.
        :type qty: int
        :param filled: How many shares of the order have already been filled, if any.
        :type filled: int
        :param commission: The amount of commission associated with placing the order.
        :type commission: int
        :param created: The date and time that the order was created
        :type created: datetime
        :raises NotAnAssetError, InvalidActionError:

        NOTES
        -----
        See :class:`pytech.enums.OrderType` to see valid order types
        See :class:`pytech.enums.OrderSubType` to see valid order sub types
        """

        if issubclass(asset.__class__, Asset):
            self.asset = asset
        else:
            raise NotAnAssetError('asset must be an instance of a subclass of the Asset class. {} was provided'
                                  .format(type(asset)))

        self.portfolio = portfolio
        # TODO: validate that all of these inputs make sense together. e.g. if its a stop order stop shouldn't be none
        self.action = TradeAction.check_if_valid(action)
        self.order_type = OrderType.check_if_valid(order_type)
        self.order_subtype = OrderSubType.check_if_valid(order_subtype)

        if self.action is TradeAction.SELL:
            if qty > 0:
                self.qty = qty * -1
            else:
                self.qty = qty
        else:
            self.qty = qty

        self.commission = commission
        self.stop = stop
        self.limit = limit
        self.stop_reached = False
        self.limit_reached = False
        self.filled = filled
        self._status = OrderStatus.OPEN
        self.reason = None
        self.created = utils.parse_date(created)
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
    def triggered(self):
        if self.stop is not None and not self.stop_reached:
            return False

        if self.limit is not None and not self.limit_reached:
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

    def check_order_triggers(self):
        """
        Check if any of the order's triggers should be pulled and execute a trade and then delete the order.
        :return:
        :rtype:
        """



        if self.triggered:
            return


class Trade(Base):
    """
    This class is used to make trades and keep trade of past trades
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

    # owned_stock_id = Column(Integer, ForeignKey('owned_stock.id'))
    # owned_stock = relationship('OwnedStock')
    # corresponding_trade = relationship('Trade', remote_side=[id])

    def __init__(self, qty, price_per_share, strategy, action, trade_date=None, ticker=None):
        """
        :param trade_date: datetime, corresponding to the date and time of the trade date
        :param qty: int, number of shares traded
        :param price_per_share: float
            price per individual share in the trade or the average share price in the trade
        :param ticker:
            a :class: Stock, the asset object that was traded
        :param action: str, must be *buy* or *sell* depending on what kind of trade it was
        :param position: str, must be *long* or *short*
        """
        if trade_date:
            self.trade_date = utils.parse_date(trade_date)
        else:
            self.trade_date = datetime.now()

        self.action = TradeAction.check_if_valid(action)
        self.strategy = strategy.lower()
        self.ticker = ticker
        self.qty = qty
        self.price_per_share = price_per_share
        self.corresponding_trade_id = self._get_corresponding_trade_id(ticker=ticker)

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

import logging
from abc import ABCMeta
from pytech.utils.enums import EventType, TradeAction, OrderType, Position, \
    SignalType


class Event(metaclass=ABCMeta):
    """
    Base Class that all Events must inherit from.

    Provides an interface for which all events are handled.
    """

    LOGGER_NAME = 'EVENT'

    def __init__(self):
        self.logger = logging.getLogger(self.LOGGER_NAME)


class MarketEvent(Event):
    """Handles the event of receiving new market data."""

    LOGGER_NAME = EventType.MARKET.name

    def __init__(self):
        super().__init__()
        self.type = EventType.MARKET


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a :class:`Strategy`.
    Which is received by a :class:`Portfolio` and acted upon.
    """

    LOGGER_NAME = EventType.SIGNAL.name

    def __init__(self, ticker, dt, signal_type):
        """
        Base SignalEvent constructor.
        
        :param ticker: The ticker to create the signal for.
        :param dt: The date the signal is being created. 
        :param signal_type: The type of signal being created.
        """

        super().__init__()

        self.type = EventType.SIGNAL
        self.ticker = ticker
        self.dt = dt
        self.signal_type = SignalType.check_if_valid(signal_type)


class HoldSignalEvent(SignalEvent):
    """Indicates a hold signal."""
    SIGNAL_TYPE = SignalType.HOLD

    def __init__(self, ticker, dt):
        super().__init__(ticker, dt, self.SIGNAL_TYPE)


class ExitSignalEvent(SignalEvent):
    """
    Signal indicating to close all positions for an Asset.
    
    An :class:`OrderType` as well as a stop and/or limit price may be specified 
    otherwise a MarketOrder will be created.
    """

    SIGNAL_TYPE = SignalType.EXIT

    def __init__(self, ticker, dt, order_type=OrderType.MARKET,
                 stop_price=None, limit_price=None):
        """
        Constructor for ExitSignal.
        
        :param order_type: (optional) The :class:`OrderType` to create to 
        exit the position.
        (default: ``OrderType.MARKET``)
        :param stop_price: (optional) The stop price for the order.
        If the order type is not ``OrderType.STOP`` or ``OrderType.STOP_LIMIT``
        then this value is ignored.
        :param limit_price: (optional) The limit price for the order.
        If the order type is not ``OrderType.LIMIT`` 
        or ``OrderType.STOP_LIMIT`` then this value is ignored. 
        """
        super().__init__(ticker, dt, self.SIGNAL_TYPE)
        self.order_type = OrderType.check_if_valid(order_type)
        self.stop_price = stop_price
        self.limit_price = limit_price


class CancelSignalEvent(SignalEvent):
    """
    Signal indicating that all open orders for a given asset should be 
    canceled.
    """
    SIGNAL_TYPE = SignalType.CANCEL

    def __init__(self, ticker, dt, trade_action=None, upper_price=None,
                 lower_price=None, order_type=None):
        """
        Constructor for CancelSignal.
        
        :param TradeAction trade_action: (optional) Type of open orders to 
        cancel. If None, then all orders will be canceled.
        :param float upper_price: (optional) Only cancel orders that are for 
        more than this amount.
        :param float lower_price: (optional) Only cancel orders that are below
        this this amount
        :param OrderType order_type: (optional) Only cancel orders that are of 
        the provided the order_type. e.g. Only cancel stop orders.
        """
        super().__init__(ticker, dt, self.SIGNAL_TYPE)
        self.trade_action = TradeAction.check_if_valid(trade_action)
        self.upper_price = upper_price
        self.lower_price = lower_price
        self.order_type = OrderType.check_if_valid(order_type)


class TradeSignalEvent(SignalEvent):
    """
    Signal indicating a possible trade opportunity. Either long or short.
    
    These events will **ALWAYS** generate a new order and will not affect any 
    other order that already exists.
    
    Most of the attributes are optional because it is up to the portfolio to 
    determine how to interpret them.
    """
    SIGNAL_TYPE = SignalType.TRADE

    def __init__(self, ticker, dt, position, order_type, strength=None,
                 stop_price=None, limit_price=None, target_price=None):
        """
        Constructor for TradeSignalEvent
        
        :param Position position: LONG or SHORT
        :param OrderType order_type: Type of order to create
        :param Any strength: An indicator about confidence in the signal.
        :param stop_price: (optional) The stop price.
        :param limit_price: (optional) The limit price. 
        :param target_price: (optional) The ideal price to buy at.
        """
        super().__init__(ticker, dt, self.SIGNAL_TYPE)
        self.position = Position.check_if_valid(position)
        self.order_type = OrderType.check_if_valid(order_type)
        self.strength = strength
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.target_price = target_price


class TradeEvent(Event):
    """
    Handles the event of actually executing an order and sending it to a broker 
    for execution/filling.

    This event should only be created in the event that an :class:`Order` 
    is triggered.
    """
    LOGGER_NAME = EventType.TRADE.name

    def __init__(self, order_id, price, qty, dt):
        super().__init__()
        self.type = EventType.TRADE
        self.order_id = order_id
        self.price = price
        self.qty = qty
        self.dt = dt

    def log_order(self):
        self.logger.info('Order: ')


class FillEvent(Event):
    """
    Handles the event of a broker actually filling the order and returning 
    either cash or the asset.
    """
    LOGGER_NAME = EventType.FILL.name

    def __init__(self, order_id, price, available_volume, dt):
        super().__init__()
        self.type = EventType.FILL
        self.order_id = order_id
        self.price = price
        self.available_volume = available_volume
        self.dt = dt

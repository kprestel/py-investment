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

    def __init__(self, ticker, dt, signal_type, strength=None,
                 target_price=None):
        super().__init__()

        self.type = EventType.SIGNAL
        self.ticker = ticker
        self.dt = dt
        # long, short, exit, hold, or cancel
        # TODO: hold all orders
        self.signal_type = SignalType.check_if_valid(signal_type)
        # a way of adding extra logic?
        self.strength = strength
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

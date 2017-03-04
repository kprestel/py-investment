import logging
from abc import ABCMeta
from pytech.utils.enums import EventType, TradeAction, OrderType, Position


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

    def __init__(self, ticker, dt, signal_type, strength):

        super().__init__()

        self.type = EventType.SIGNAL
        self.ticker = ticker
        self.dt = dt
        # long or short
        self.signal_type = TradeAction.check_if_valid(signal_type)
        self.strength = strength


class OrderEvent(Event):

    LOGGER_NAME = EventType.ORDER.name

    def __init__(self, ticker, order_type, qty, action):

        super().__init__()
        self.type = EventType.ORDER
        self.ticker = ticker
        self.order_type = OrderType.check_if_valid(order_type)
        self.qty = qty
        self.action = TradeAction.check_if_valid(action)

    def log_order(self):

        self.logger.info('Order: ')


class FillEvent(Event):

    LOGGER_NAME = EventType.FILL.name

    def __init__(self, time_index, ticker, exchange, qty, action, fill_cost, commission=None):

        super().__init__()
        self.type = EventType.FILL
        self.time_index = time_index
        self.ticker = ticker
        self.exchange = exchange
        self.qty = qty
        self.action = TradeAction.check_if_valid(action)
        self.fill_cost = fill_cost
        # TODO: make this use commission models.
        self.commission = commission


import datetime
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

import pytech.utils.dt_utils as dt_utils
from pytech.utils.enums import (EventType, OrderType, Position, SignalType,
                                TradeAction)


class Event(metaclass=ABCMeta):
    """
    Base Class that all Events must inherit from.

    Provides an interface for which all events are handled.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Must return the classes :class:``EventType``"""

    @classmethod
    def from_dict(cls, event_dict: Dict):
        return cls(**{k: v for k, v in event_dict.items()})

    @classmethod
    def get_subclasses(cls):
        yield cls.__subclasses__() + [y for x in cls.__subclasses__()
                                      for y in x.get_subclasses()]


class MarketEvent(Event):
    """Handles the event of receiving new market data."""

    def __init__(self):
        super().__init__()

    @property
    def event_type(self) -> EventType:
        return EventType.MARKET


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a :class:`Strategy`.
    Which is received by a :class:`Portfolio` and acted upon.
    """

    def __init__(self,
                 ticker: str,
                 dt: datetime or str,
                 signal_type: SignalType or str,
                 limit_price: float = None,
                 stop_price: float = None,
                 target_price: float = None,
                 strength: float = None,
                 order_type: OrderType = None,
                 action: TradeAction = None,
                 position: Position = None,
                 upper_price: float = None,
                 lower_price: float = None,
                 *args,
                 **kwargs):
        """
        Base SignalEvent constructor.
        
        Basic rules:
            - If stop and limit price are ``None`` a :class:``MarketOrder``
            will be created.
            - If only stop price is provided a :class:``StopOrder`` would be
            created.
            - If only limit price is provided a :class:``LimitPrice`` would be
            created.
            - If both stop and limit price are provided a 
        :param upper_price: 
        :param lower_price: 
        :param position: 
        :param action: 
        :param limit_price: 
        :param stop_price: 
        :param target_price: 
        :param strength: 
        :param order_type: 
        :class:``StopLimitOrder`` would be created.
        - ``target_price`` and ``strength`` **DO NOT** have any sort of
        :param ticker: The ticker to create the signal for.
        :param dt: The date the signal is being created. 
        :param signal_type: The type of signal being created.
        """

        super().__init__()

        self.ticker = ticker
        self.dt = dt_utils.parse_date(dt)
        self.signal_type = SignalType.check_if_valid(signal_type)
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.strength = strength
        self.upper_price = upper_price
        self.lower_price = lower_price

        if action is not None:
            self.action = TradeAction.check_if_valid(action)
        else:
            self.action = action

        if position is not None:
            self.position = Position.check_if_valid(position)
        else:
            self.position = None

        if order_type is not None:
            self.order_type = OrderType.check_if_valid(order_type)
        else:
            self.order_type = None

    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL


class TradeEvent(Event):
    """
    Handles the event of actually executing an order and sending it to a broker 
    for execution/filling.

    This event should only be created in the event that an :class:`Order` 
    is triggered.
    """

    def __init__(self,
                 order_id: str,
                 price: float,
                 qty: int,
                 dt: datetime or str):
        super().__init__()
        self.order_id = order_id
        self.price = price
        self.qty = qty
        self.dt = dt_utils.parse_date(dt)

    @property
    def event_type(self):
        return EventType.TRADE


class FillEvent(Event):
    """
    Handles the event of a broker actually filling the order and returning 
    either cash or the asset.
    """

    def __init__(self,
                 order_id: str,
                 price: float,
                 available_volume: int,
                 dt: datetime or str):
        super().__init__()
        self.order_id = order_id
        self.price = price
        self.available_volume = available_volume
        self.dt = dt_utils.parse_date(dt)

    @property
    def event_type(self):
        return EventType.FILL

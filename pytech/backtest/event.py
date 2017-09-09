import datetime
import logging
from abc import (
    ABCMeta,
    abstractmethod
)
from typing import (
    Any,
    Dict,
    Union
)

import pytech.utils.dt_utils as dt_utils
from pytech.utils.enums import (
    EventType,
    OrderType,
    Position,
    SignalType,
    TradeAction
)


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
                 signal_type: Union[SignalType, str],
                 limit_price: float = None,
                 stop_price: float = None,
                 target_price: float = None,
                 strength: Any = None,
                 order_type: OrderType = OrderType.MARKET,
                 action: TradeAction = None,
                 position: Position = None,
                 upper_price: float = None,
                 lower_price: float = None,
                 *args,
                 **kwargs):
        """
        Base SignalEvent constructor.

        Basic rules:
            * If stop and limit price are ``None`` a :class:`MarketOrder`
            will be created.

            * If only stop price is provided a :class:`StopOrder` would be
            created.

            * If only limit price is provided a :class:`LimitOrder` would be
            created.

            * If both stop and limit price are provided a
            :class:`StopLimitOrder` will be created

            * If ``signal_type`` is simply ``TRADE`` then it could be
            considered ``LONG`` of ``SHORT`` it is up to the portfolio
            to determine this.

        :param upper_price:
        :param lower_price:
        :param position:
        :param action:
        :param limit_price:
        :param stop_price:
        :param target_price:
        :param strength:
        :param order_type:
        :class:`StopLimitOrder` would be created.
        - ``target_price`` and ``strength`` **DO NOT** have any sort of
        :param ticker: The ticker to create the signal for.
        :param signal_type: The type of signal being created.
        """

        super().__init__()

        self.ticker = ticker
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

        if order_type is OrderType.MARKET:
            # try to determine what kind of order to place
            if stop_price is None and limit_price is not None:
                order_type = OrderType.LIMIT
            elif stop_price is not None and limit_price is None:
                order_type = OrderType.STOP
            elif stop_price is not None and limit_price is not None:
                order_type = OrderType.STOP_LIMIT
            else:
                order_type = OrderType.MARKET
                # Typically a Market order is not desirable so we warn on it.
                self.logger.warning(
                    'Creating a SignalEvent with a Market order type.')

        self.order_type = OrderType.check_if_valid(order_type)

    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL


class TradeSignalEvent(SignalEvent):
    """A SignalEvent that is specifically for ``LONG`` or ``SHORT`` trades."""

    def __init__(self,
                 ticker: str,
                 signal_type: SignalType or str,
                 limit_price: float = None,
                 stop_price: float = None,
                 target_price: float = None,
                 strength: Any = None,
                 order_type: OrderType = None,
                 action: TradeAction = None,
                 position: Position = None,
                 upper_price: float = None,
                 lower_price: float = None,
                 *args, **kwargs):
        super().__init__(ticker,
                         signal_type,
                         limit_price,
                         stop_price,
                         target_price,
                         strength,
                         order_type,
                         action,
                         position,
                         upper_price,
                         lower_price,
                         *args, **kwargs)


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
    def event_type(self) -> EventType:
        return EventType.FILL

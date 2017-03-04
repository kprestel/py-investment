from enum import Enum

from pytech.utils.exceptions import InvalidActionError, InvalidOrderStatusError, InvalidOrderSubTypeError, \
    InvalidOrderTypeError, InvalidPositionError


class AutoNumber(Enum):
    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    @classmethod
    def check_if_valid(cls, value):
        if value is None:
            return None
        elif isinstance(value, cls):
            return value
        for name, member in cls.__members__.items():
            if member.name == value.upper():
                return member
        else:
            return None


class EventType(AutoNumber):
    """Different Event types"""

    MARKET = ()
    SIGNAL = ()
    ORDER = ()
    FILL = ()


class SignalType(AutoNumber):
    BUY = ()
    SELL = ()
    EXIT = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidActionError(action=value)


class TradeAction(AutoNumber):
    BUY = ()
    SELL = ()
    EXIT = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidActionError(action=value)


class OrderStatus(AutoNumber):
    OPEN = ()
    FILLED = ()
    CANCELLED = ()
    REJECTED = ()
    HELD = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidOrderStatusError(order_status=value)


class OrderType(AutoNumber):
    """Valid Order Types"""
    # TODO: document this better
    STOP = ()
    LIMIT = ()
    STOP_LIMIT = ()
    MARKET = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidOrderTypeError(order_type=value)


class OrderSubType(AutoNumber):
    """Valid OrderSubtypes"""
    ALL_OR_NONE = ()
    GOOD_TIL_CANCELED = ()
    DAY = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidOrderSubTypeError(order_subtype=value)


class Position(AutoNumber):
    LONG = ()
    SHORT = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidPositionError(position=value)

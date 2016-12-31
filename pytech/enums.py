from enum import Enum
from pytech.exceptions import PyInvestmentError, InvalidActionError, InvalidPositionError, InvalidOrderStatusError, \
    InvalidOrderTypeError


class AutoNumber(Enum):
    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    @classmethod
    def check_if_valid(cls, value):
        if isinstance(value, cls):
            return value.name
        for name, member in cls.__members__.items():
            if member.name == value.upper():
                return member.name
        else:
            return None

class TradeAction(AutoNumber):
    BUY = ()
    SELL = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidActionError('action must either be "BUY" or "SELL". {} was provided'.format(value))

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
            raise InvalidOrderStatusError('order_status must either be "OPEN", "FILLED", "CANCELLED", "REJECTED", or "HELD". {} was provided'
                                          .format(value))

class OrderType(AutoNumber):
    STOP = ()
    LIMIT = ()
    STOP_LIMIT = ()
    MARKET = ()
    ALL_OR_NONE = ()
    GOOD_TIL_CANCELED = ()
    DAY = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidOrderTypeError('order_type must either be "STOP", "LIMIT", "STOP_LIMIT", "MARKET", "ALL_OR_NONE", '
                                        '"GOOD_TILL_CANCELED", "DAY". {} was provided'.format(value))


class AssetPosition(AutoNumber):
    LONG = ()
    SHORT = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise InvalidPositionError('action must either be "BUY" or "SELL". {} was provided'.format(value))

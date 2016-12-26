from enum import Enum
from pytech.exceptions import PyInvestmentError, InvalidActionError, InvalidPositionError


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

class TradeActions(AutoNumber):
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

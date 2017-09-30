from typing import TypeVar

from .order import (
    Order,
    StopLimitOrder,
    LimitOrder,
    StopOrder,
    MarketOrder
)


def get_order_types() -> TypeVar:
    """Return valid order types for type annotations."""
    return TypeVar('A',
                   Order,
                   MarketOrder,
                   StopOrder,
                   LimitOrder,
                   StopLimitOrder)


# AnyOrder = get_order_types()
AnyOrder = TypeVar('A',
                   Order,
                   MarketOrder,
                   StopOrder,
                   LimitOrder,
                   StopLimitOrder)

from .controls import TradingControl
from .blotter import Blotter

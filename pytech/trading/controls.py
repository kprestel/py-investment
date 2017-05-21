import datetime as dt
import logging
from abc import ABCMeta, abstractmethod
from typing import Any

from pytech.utils.exceptions import TradeControlViolation


class TradingControl(metaclass=ABCMeta):
    """
    ABC class representing fail-safe control on the behavior of any algorithm.
    """

    def __init__(self, raise_on_error, **kwargs):
        self.raise_on_violation = raise_on_error
        self.__fail_args = kwargs
        self.logger = logging.getLogger(__name__)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__fail_args})'

    @abstractmethod
    def validate(self,
                 ticker: str,
                 qty: int,
                 bt_dt: dt.datetime,
                 current_price: float = None) -> None:
        """
        Before any :class:`pytech.trading.order.Order` is placed by the 
        :class:`Blotter` this method *must* be called *exactly once* for each
        registered :class:`TradingControl` object.
        
        If the specified constraint is *not* violated then this method should 
        return `None` and have no externally visible side-effects.
        
        If the specified constraint *is* violated then this method *must* call
        `self.fail(ticker, qty, dt)`
        
        :param ticker: The ticker that is being traded.
        :param qty: The amount of shares being traded.
        :param bt_dt: The current date in the backtest.
        :param current_price: The current price of the asset.
        :raises TradeControlViolation: If the trade control is violated.
        """
        raise NotImplementedError

    def _constraint_msg(self, metadata: Any) -> str:
        """Create the error message."""
        constraint = repr(self)

        if metadata is not None:
            constraint = f'{constraint} (Metadata: {metadata})'

        return constraint

    def fail(self,
             ticker: str,
             qty: int,
             current_dt: dt.datetime,
             metadata: Any = None) -> None:
        """
        Handle a `TradingControlViolation` either by raising or logging an 
        error with information about the violation. 
        
        Whether or not an exception is raised depends on if 
        `raise_on_violation` is `True` or `False`
        
        :param ticker: The ticker that caused the violation.
        :param qty: The amount of shares that caused the violation.
        :param current_dt: The datetime that the violation occurred.
        :param metadata: Any other information that should be displayed.
        :raises TradeControlViolation: If `raise_on_violation` is `True` and
            a violation has occurred.
        """
        constraint = self._constraint_msg(metadata)
        if self.raise_on_violation:
            raise TradeControlViolation(ticker=ticker,
                                        qty=qty,
                                        dt=current_dt,
                                        constraint=constraint)
        else:
            self.logger.error(
                    f'Order for {qty} shares of {ticker} '
                    f'at {dt} violates trading constraint {constraint}')


class MaxOrderCount(TradingControl):
    """
    Trading control that represents the maximum number of orders that can be
    placed in a given trading day.
    """

    def __init__(self, raise_on_error, max_count):
        super().__init__(raise_on_error, max_count=max_count)
        self.orders_placed = 0
        self.max_count = max_count
        self.current_date = None

    def validate(self, ticker: str, qty: int, bt_dt: dt.datetime,
                 current_price: float = None) -> None:
        """Fail if we've already placed `self.max_orders` today."""
        bt_date = bt_dt.date()

        if self.current_date is not None and self.current_date != bt_date:
            self.orders_placed = 0
        self.current_date = bt_date

        if self.orders_placed > self.max_count:
            self.fail(ticker, qty, bt_dt)

        self.orders_placed += 1


class MaxOrderSize(TradingControl):
    """
    Trading control that represents a limit on the magnitude of any single 
    order placed on a given asset.
    
    Can be specified by price per share or total dollar value.
    """

    def __init__(self,
                 raise_on_error: bool,
                 ticker: str = None,
                 max_notional: float = None,
                 max_share: float = None):
        super().__init__(raise_on_error,
                         ticker=ticker,
                         max_notional=max_notional,
                         max_share=max_notional)

        if max_share is None and max_notional is None:
            raise ValueError('Must supply at least one of max_share and '
                             'max_notional.')
        if max_share is not None and max_share <= 0:
            raise ValueError(f'max_share must be greater than 0.'
                             f'{max_share} was given.')

        if max_notional is not None and max_notional <= 0:
            raise ValueError(f'max_notional must be greater than 0.'
                             f'{max_notional} was given.')

        self.max_share = max_share
        self.max_notional = max_notional
        self.ticker = ticker

    def validate(self, ticker: str, qty: int, bt_dt: dt.datetime,
                 current_price: float = None) -> None:
        if self.ticker is not None and self.ticker != ticker:
            return None

        if self.max_share is not None and abs(qty) > self.max_share:
            self.fail(ticker, qty, bt_dt)

        order_value = qty * current_price
        too_much_value = (self.max_notional is not None and
                          abs(order_value) > self.max_notional)
        if too_much_value:
            self.fail(ticker, qty, bt_dt,
                      metadata=f'order_value={order_value}')

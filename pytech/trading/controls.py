"""
This module contains any control that dictates whether or not an order goes
through.
"""
import datetime as dt
import logging
from typing import (
    Any,
    Dict,
TYPE_CHECKING
)

from abc import (
    ABCMeta,
    abstractmethod
)


if TYPE_CHECKING:
    from pytech.trading import AnyOrder
    from fin.portfolio import Portfolio
from pytech.utils.exceptions import TradingControlViolation


class TradingControl(metaclass=ABCMeta):
    """
    ABC representing a fail-safe control on the execution of a strategy.
    """

    def __init__(self, raise_on_error: bool = True, **kwargs):
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.raise_on_error: bool = raise_on_error
        self.__fail_args: Dict[str, Any] = kwargs

    @abstractmethod
    def validate(self,
                 order: 'AnyOrder',
                 portfolio: 'Portfolio',
                 cur_dt: dt.datetime,
                 price: float):
        """
        This method should be called *exactly once* on each registered
        :class:`TradingControl` object.

        If the order does not violate the control then there should be no
        visible side effects.

        If the order *does* violate the control's constraint then `self.fail`
        should be called.

        :param asset:
        :param cost:
        :param portfolio:
        :param cur_dt:
        :param price:
        :return:
        """
        raise NotImplementedError

    def _control_msg(self, **kwargs) -> str:
        """
        Construct the failure message to explain to user how and what
        happened.

        Any extra arguments may be passed in via `kwargs`.

        :param kwargs:
        :return:
        """
        msg = repr(self)
        if kwargs is None:
            return msg

        for k, v in kwargs.items():
            msg += f'{k}: {v} '
        return msg

    def _fail(self, order: 'AnyOrder', cur_dt: dt.datetime, **kwargs):
        """
        Handle a :class:`TradingConstraint` violation.

        Any extra arguments may be passed in via `kwargs` that should be
        displayed to the user.

        :param order: the order that caused the failure.
        :param cur_dt: the current date in the algorithm.
        :return:
        """
        control = self._control_msg()

        if self.raise_on_error:
            raise TradingControlViolation(qty=order.qty,
                                          ticker=order.ticker,
                                          datetime=cur_dt,
                                          control=control)
        else:
            self.logger.error(
                f'Order for {order.qty} shares of {order.ticker} at {cur_dt} '
                f'violates trading control {control}.')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__fail_args})'


class MaxOrderSize(TradingControl):
    """
    Trading control representing a limit on the magnitude of any single order
    placed with the given asset.

    This can be in terms of number of shares or a dollar amount.
    """

    def __init__(self, raise_on_error: bool, ticker: str,
                 max_shares: int = None, max_notional: float = None):
        super().__init__(raise_on_error,
                         ticker=ticker,
                         max_shares=max_shares,
                         max_notional=max_notional)
        self.ticker = ticker
        if max_shares is None and max_notional is None:
            raise ValueError('Must supply at least one of max_shares,'
                             'max_notional, or max_pct.')

        if max_shares is not None and max_shares < 0:
            raise ValueError('max_shares must be positive.')

        if max_notional is not None and max_notional < 0:
            raise ValueError('max_notional must be positive.')

        self.max_shares: int = max_shares
        self.max_notional: float = max_notional

    def validate(self,
                 order: 'AnyOrder',
                 portfolio: 'Portfolio',
                 cur_dt: dt.datetime,
                 price: float) -> None:
        """
        Fail if the given order would cause the total position (current +
        order) to be greater than `self.max_shares` or a greater than
        `self.max_notional`.

        :param order: the proposed order.
        :param portfolio: the portfolio that the order is being placed for.
        :param cur_dt: the current date in the algorithm.
        :param price: the current price of the asset.
        """
        if self.ticker != order.ticker:
            return

        try:
            current_shares = portfolio.owned_assets[order.ticker].shares_owned
        except KeyError:
            current_shares = 0

        shares_post_order = current_shares + order.qty

        too_many_shares = (self.max_shares is not None
                           and abs(shares_post_order) > self.max_shares)

        if too_many_shares:
            self._fail(order, cur_dt)

        value_post_order = shares_post_order * price

        too_much_value = (self.max_notional is not None and
                          abs(value_post_order) > self.max_notional)

        if too_much_value:
            self._fail(order, cur_dt)


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

    def validate(self, order: 'AnyOrder',
                 portfolio: 'Portfolio',
                 cur_dt: dt.datetime,
                 price: float) -> None:
        """Fail if we've already placed `self.max_orders` today."""
        bt_date = cur_dt.date()

        if self.current_date is not None and self.current_date != bt_date:
            self.orders_placed = 0
        self.current_date = bt_date

        if self.orders_placed > self.max_count:
            self._fail(order, cur_dt)

        self.orders_placed += 1

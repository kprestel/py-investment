import logging
import math
from abc import ABCMeta, abstractmethod
from typing import List, TYPE_CHECKING

import pytech.utils as utils
from pytech.exceptions import InsufficientFundsError, InvalidSignalTypeError
from pytech.backtest.event import SignalEvent
from pytech.utils.enums import SignalType, TradeAction

if TYPE_CHECKING:
    from pytech.fin.portfolio import Portfolio
    from pytech.trading.blotter import AnyOrder


class SignalHandler(metaclass=ABCMeta):
    """ABC for Signal Handlers."""

    def __init__(self, raise_on_warnings: bool = True):
        """
        Constructor for the `SignalHandler` base classes
        :param raise_on_warnings: if true exceptions will be raised, otherwise
            they will just be logged as warnings. The main purpose of this
            attribute is to determine whether or not to raise an
            :class:``InsufficientFundsError`` when one is encountered.

        """
        self.logger = logging.getLogger(__name__)
        self.raise_on_warnings = raise_on_warnings

    def handle_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        """
        Must handle a signal event.

        Possible ways to handle a signal:
            * Place a :class:``Order``

                * This could be a :class:``MarketOrder``
                * Or a :class:``StopOrder``
                * Or any kind of :class:``Order`` depending on the signal and
                the state of the portfolio.

            * Ignore it, if it does not make sense to act upon it based on the
            state of the ``portfolio``
        """
        if not isinstance(signal, SignalEvent):
            raise TypeError(
                "signal must be a SignalEvent. " f"{type(signal)} was provided."
            )

        if signal.signal_type is SignalType.EXIT:
            self._handle_exit_signal(portfolio, triggered_orders, signal)
        elif signal.signal_type is SignalType.CANCEL:
            self._handle_cancel_signal(portfolio, triggered_orders, signal)
        elif signal.signal_type is SignalType.HOLD:
            self._handle_hold_signal(portfolio, triggered_orders, signal)
        elif signal.signal_type is SignalType.TRADE:
            self._handle_trade_signal(portfolio, triggered_orders, signal)
        elif signal.signal_type is SignalType.LONG:
            self._handle_long_signal(portfolio, triggered_orders, signal)
        elif signal.signal_type is SignalType.SHORT:
            self._handle_short_signal(portfolio, triggered_orders, signal)
        else:
            raise InvalidSignalTypeError(signal_type=type(signal.signal_type))

    def _handle_trade_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        """
        Handle a trade :class:`SignalEvent`.

        :param signal:
        :return:
        """
        try:
            if signal.position is SignalType.LONG:
                self._handle_long_signal(signal)
            elif signal.position is SignalType.SHORT:
                self._handle_short_signal(signal)
            else:
                # default always to general trade signals.
                self._handle_general_trade_signal(signal)
        except AttributeError:
            self._handle_general_trade_signal(signal)

    def _handle_hold_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        """
        Handle a ``HOLD`` :class:`SignalEvent`

        :param portfolio:
        :param signal:
        :return:
        """
        portfolio.blotter.hold_all_orders_for_asset(signal.ticker)

    def _handle_cancel_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        """
        Handle a ``CANCEL`` :class:`SignalEvent`

        :param portfolio:
        :param signal:
        :return:
        """
        portfolio.blotter.cancel_all_orders_for_asset(
            signal.ticker,
            upper_price=signal.upper_price,
            lower_price=signal.lower_price,
            order_type=signal.order_type,
        )

    def _handle_exit_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        """
        Handle an ``EXIT`` :class:`SignalEvent`.

        :param signal:
        :return:
        """
        qty = portfolio.owned_assets[signal.ticker].shares_owned

        if qty > 0:
            action = TradeAction.SELL
        elif qty < 0:
            action = TradeAction.BUY
        else:
            raise ValueError(
                f"Cannot exit from a position that is not owned. "
                f"Owned qty is 0 for ticker: {signal.ticker}."
            )

        portfolio.blotter.place_order(
            self,
            signal.ticker,
            qty,
            action,
            signal.order_type,
            signal.stop_price,
            signal.limit_price,
        )

    @abstractmethod
    def _handle_long_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        """
        Handle a ``LONG`` :class:`SignalEvent`

        :param portfolio:
        :param signal:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_short_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):

        """
        Handle a ``SHORT`` :class:`SignalEvent`.

        :param portfolio:
        :param signal:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_general_trade_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        """
        Handle an ambiguous ``TRADE`` :class:`SignalEvent`. It is up to the
        child classes to determine how to handle this.

        :param portfolio:
        :param signal:
        :return:
        """
        raise NotImplementedError


class BasicSignalHandler(SignalHandler):
    """A basic implementation of a Signal handler."""

    def __init__(
        self,
        raise_on_warnings: bool = True,
        include_cash: bool = False,
        max_weight: float = 0.25,
        min_weight: float = 0.05,
        target_weight: float = 0.15,
        price_col: str = utils.ADJ_CLOSE_COL,
    ) -> None:
        """
        Constructor for the signal handler.

        :param include_cash: should cash be considered when calculating
            portfolio weights.
        :param max_weight: the max weight a single asset should account for in
            the portfolio. This is both positive and negative, meaning a
            ``max_weight`` of ``.25`` would prevent a ``LONG`` position from
            being more than 25% of total portfolio value and a ``SHORT``
            position from being more than -25% of total portfolio value.
        :param min_weight: same as ``max_weight`` except the minimum weight an
            asset should represent in a portfolio.
        :param target_weight: the target weight that a single asset should
            account for in a portfolio.
        :param price_col: the column to get the latest price from.
        """
        super().__init__(raise_on_warnings)
        self.include_cash = include_cash

        if max_weight > 0:
            self.max_weight = max_weight
        else:
            self.max_weight = max_weight * -1

        if min_weight > 0:
            self.min_weight = min_weight
        else:
            self.min_weight = min_weight * -1

        if target_weight > 0:
            self.target_weight = target_weight
        else:
            self.target_weight = target_weight * -1

        self.price_col = price_col

    def _handle_short_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        pass

    def _handle_hold_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        super()._handle_hold_signal(portfolio, triggered_orders, signal)

    def _handle_long_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        if signal.ticker in triggered_orders:
            self.logger.info(
                f"ticker: {signal.ticker} already had an order"
                f"triggered. Not acting on signal."
            )
            return

        current_weight = portfolio.get_asset_weight(signal.ticker, self.include_cash)
        if current_weight >= self.max_weight or current_weight <= (
            self.max_weight * -1
        ):
            self.logger.info(
                f"ticker: {signal.ticker} is over the max_weight "
                f"of {self.max_weight} at a current weight of "
                f"f{current_weight}. Not acting on signal."
            )
            return

        price = portfolio.bars.latest_bar_value(signal.ticker, self.price_col)
        qty = self._get_target_qty(price, portfolio, signal)

        if portfolio.check_liquidity(price, qty):
            portfolio.blotter.place_order(portfolio, qty=qty, **signal.__dict__)
        elif self.raise_on_warnings:
            raise InsufficientFundsError(ticker=signal.ticker)
        else:
            self.logger.warning(
                f"Attempted to place an order for "
                f"{signal.ticker} but lack sufficient funds."
            )

    def _handle_general_trade_signal(
        self,
        portfolio: "Portfolio",
        triggered_orders: List["AnyOrder"],
        signal: SignalEvent,
    ):
        pass

    def _get_target_qty(
        self, price: float, portfolio: "Portfolio", signal: SignalEvent
    ) -> int:
        if self.include_cash or portfolio.total_asset_mv == 0.0:
            mv = portfolio.total_value
        else:
            mv = portfolio.total_asset_mv

        return int(math.floor((self.target_weight * mv) / price))

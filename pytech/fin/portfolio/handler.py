import logging
from abc import (
    ABCMeta,
    abstractmethod
)
from typing import List, TYPE_CHECKING

from pytech.backtest.event import SignalEvent
from pytech.utils.enums import (
    SignalType,
    TradeAction
)
from pytech.utils.exceptions import InvalidSignalTypeError
if TYPE_CHECKING:
    from pytech.fin.portfolio import Portfolio
    from trading.blotter import AnyOrder


class SignalHandler(metaclass=ABCMeta):
    """ABC for Signal Handlers."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def handle_signal(self, portfolio: 'Portfolio',
                      triggered_orders: List['AnyOrder'],
                      signal: SignalEvent):
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
            raise TypeError('signal must be a SignalEvent. '
                            f'{type(signal)} was provided.')

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

    def _handle_trade_signal(self, portfolio: 'Portfolio',
                             triggered_orders: List['AnyOrder'],
                             signal: SignalEvent):
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

    def _handle_hold_signal(self, portfolio: 'Portfolio',
                            triggered_orders: List['AnyOrder'],
                            signal: SignalEvent):
        """
        Handle a ``HOLD`` :class:`SignalEvent`

        :param portfolio:
        :param signal:
        :return:
        """
        portfolio.blotter.hold_all_orders_for_asset(signal.ticker)

    def _handle_cancel_signal(self, portfolio: 'Portfolio',
                              triggered_orders: List['AnyOrder'],
                              signal: SignalEvent):
        """
        Handle a ``CANCEL`` :class:`SignalEvent`

        :param portfolio:
        :param signal:
        :return:
        """
        portfolio.blotter.cancel_all_orders_for_asset(signal.ticker,
                                                      upper_price=signal.upper_price,
                                                      lower_price=signal.lower_price,
                                                      order_type=signal.order_type)

    def _handle_exit_signal(self, portfolio: 'Portfolio',
                            triggered_orders: List['AnyOrder'],
                            signal: SignalEvent):
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
                f'Cannot exit from a position that is not owned. '
                f'Owned qty is 0 for ticker: {signal.ticker}.')

        portfolio.blotter.place_order(self,
                                      signal.ticker,
                                      qty,
                                      action,
                                      signal.order_type,
                                      signal.stop_price,
                                      signal.limit_price)

    @abstractmethod
    def _handle_long_signal(self, portfolio: 'Portfolio',
                            triggered_orders: List['AnyOrder'],
                            signal: SignalEvent):
        """
        Handle a ``LONG`` :class:`SignalEvent`

        :param portfolio:
        :param signal:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_short_signal(self, portfolio: 'Portfolio',
                             triggered_orders: List['AnyOrder'],
                             signal: SignalEvent):

        """
        Handle a ``SHORT`` :class:`SignalEvent`.

        :param portfolio:
        :param signal:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_general_trade_signal(self, portfolio: 'Portfolio',
                                     triggered_orders: List['AnyOrder'],
                                     signal: SignalEvent):
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

    def __init__(self):
        """Constructor for BasicStrategyHandler"""
        super().__init__()

    def _handle_short_signal(self, portfolio: 'Portfolio',
                             triggered_orders: List['AnyOrder'],
                             signal: SignalEvent):
        pass

    def _handle_hold_signal(self, portfolio: 'Portfolio',
                            triggered_orders: List['AnyOrder'],
                            signal: SignalEvent):
        super()._handle_hold_signal(portfolio, triggered_orders, signal)

    def _handle_long_signal(self, portfolio: 'Portfolio',
                            triggered_orders: List['AnyOrder'],
                            signal: SignalEvent):
        pass

    def _handle_general_trade_signal(self, portfolio: 'Portfolio',
                                     triggered_orders: List['AnyOrder'],
                                     signal: SignalEvent):
        pass

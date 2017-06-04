import logging
from abc import ABCMeta, abstractmethod

import pandas as pd

import pytech.utils.pandas_utils as pd_utils
from pytech.backtest.event import SignalEvent
from pytech.fin.portfolio import AbstractPortfolio
from pytech.utils.enums import SignalType, TradeAction
from pytech.utils.exceptions import InvalidSignalTypeError


class AbstractSignalHandler(metaclass=ABCMeta):
    """ABC for Signal Handlers."""

    def __init__(self, portfolio: AbstractPortfolio):
        self.logger = logging.getLogger(__name__)
        self.portfolio = portfolio
        # easier access to the blotter
        self.blotter = portfolio.blotter
        # easier access to the bars
        self.bars = portfolio.bars

    def handle_signal(self, signal: SignalEvent):
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
        if signal.signal_type is SignalType.EXIT:
            self._handle_exit_signal(signal)
        elif signal.signal_type is SignalType.CANCEL:
            self._handle_cancel_signal(signal)
        elif signal.signal_type is SignalType.HOLD:
            self._handle_hold_signal(signal)
        elif signal.signal_type is SignalType.TRADE:
            self._handle_trade_signal(signal)
        else:
            raise InvalidSignalTypeError(signal_type=type(signal.signal_type))

    def _handle_trade_signal(self, signal: SignalEvent):
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

    def _handle_hold_signal(self, signal: SignalEvent):
        """
        Handle a ``HOLD`` :class:`SignalEvent`

        :param portfolio:
        :param signal:
        :return:
        """
        self.blotter.hold_all_orders_for_asset(signal.ticker)

    def _handle_cancel_signal(self, signal: SignalEvent):
        """
        Handle a ``CANCEL`` :class:`SignalEvent`

        :param signal:
        :return:
        """
        self.blotter.cancel_all_orders_for_asset(signal.ticker,
                                                 upper_price=signal.upper_price,
                                                 lower_price=signal.lower_price,
                                                 order_type=signal.order_type)

    def _handle_exit_signal(self, signal: SignalEvent):
        """
        Handle an ``EXIT`` :class:`SignalEvent`.

        :param signal:
        :return:
        """
        qty = self.portfolio.owned_assets[signal.ticker].shares_owned

        if qty > 0:
            action = TradeAction.SELL
        elif qty < 0:
            action = TradeAction.BUY
        else:
            raise ValueError(
                    f'Cannot exit from a position that is not owned. '
                    f'Owned qty is 0 for ticker: {signal.ticker}.')

        self.blotter.place_order(signal.ticker,
                                 qty, action,
                                 signal.order_type,
                                 signal.stop_price, signal.limit_price)

    def get_correlation_df(self, col: str = pd_utils.ADJ_CLOSE_COL,
                           market_ticker: str = 'SPY') -> pd.DataFrame:
        """
        Get the correlation between all tickers in the universe.

        :param col: The column to use in the df.
        :param market_ticker: The market ticker to add to the df.
        :return: A df with the correlation coefficients.
        """
        df = self.bars.make_agg_df(col, market_ticker)
        return df.corr()

    @abstractmethod
    def _handle_long_signal(self,
                            signal: SignalEvent):
        """
        Handle a ``LONG`` :class:`SignalEvent`

        :param signal:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_short_signal(self, signal: SignalEvent):

        """
        Handle a ``SHORT`` :class:`SignalEvent`.

        :param signal:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_general_trade_signal(self, signal: SignalEvent):
        """
        Handle an ambiguous ``TRADE`` :class:`SignalEvent`. It is up to the
        child classes to determine how to handle this.

        :param signal:
        :return:
        """
        raise NotImplementedError


class BasicSignalHandler(AbstractSignalHandler):
    """A basic implementation of a Signal handler."""

    def __init__(self, portfolio):
        """Constructor for BasicStrategyHandler"""
        super().__init__(portfolio)

    def _handle_short_signal(self, signal: SignalEvent):
        pass

    def _handle_hold_signal(self, signal: SignalEvent):
        super()._handle_hold_signal(signal)

    def _handle_long_signal(self, signal: SignalEvent):
        pass

    def _handle_general_trade_signal(self, signal: SignalEvent):
        pass

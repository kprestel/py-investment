import logging
import math
from abc import ABCMeta, abstractmethod
from typing import Dict, TYPE_CHECKING

import pytech.utils.pandas_utils as pd_utils
from pytech.backtest.event import SignalEvent
from pytech.utils.enums import SignalType, TradeAction
from pytech.utils.exceptions import InvalidSignalTypeError
from pytech.fin.portfolio import AbstractPortfolio


class AbstractBalancer(metaclass=ABCMeta):
    """Base class for all balancers."""

    def __init__(self,
                 allow_market_orders=True,
                 *args, **kwargs):
        """
        Constructor for :class:`AbstractBalancer`.
        
        :param allow_market_orders: If `True` then `EXIT` signals will be 
        allowed to execute as a market order.
        :param args: 
        :param kwargs: 
        """
        self.logger = logging.getLogger(__name__)
        self.allow_market_orders = allow_market_orders

    @abstractmethod
    def __call__(self,
                 portfolio: AbstractPortfolio,
                 signal: SignalEvent,
                 *args, **kwargs):
        """
        Balance the portfolio based on whatever methodology chosen.
        
        :param args: 
        :param kwargs: 
        :return: 
        """
        if signal.signal_type is SignalType.EXIT:
            self._handle_exit_signal(portfolio, signal)
        elif signal.signal_type is SignalType.CANCEL:
            self._handle_cancel_signal(portfolio, signal)
        elif signal.signal_type is SignalType.HOLD:
            self._handle_hold_signal(portfolio, signal)
        elif signal.signal_type is SignalType.TRADE:
            self._handle_trade_signal(portfolio, signal)
        else:
            raise InvalidSignalTypeError(signal_type=type(signal.signal_type))

    @abstractmethod
    def balance(self, portfolio: AbstractPortfolio):
        """
        Balance the portfolio based on whatever methodology chosen.
        
        :param AbstractPortfolio portfolio: 
        :return: 
        """
        raise NotImplementedError('Must implement balance(portfolio)')

    def _handle_trade_signal(self,
                             portfolio: AbstractPortfolio,
                             signal: SignalEvent):
        """
        Handle a trade :class:`SignalEvent`.
        
        :param portfolio: 
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

    @abstractmethod
    def _handle_long_signal(self,
                            signal: SignalEvent,
                            portfolio: AbstractPortfolio):
        """
        Handle a ``LONG`` :class:`SignalEvent`
        
        :param signal: 
        :param portfolio: 
        :return: 
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_short_signal(self,
                             signal: SignalEvent,
                             portfolio: AbstractPortfolio):
        """
        Handle a ``SHORT`` :class:`SignalEvent`.
        
        :param signal: 
        :param portfolio: 
        :return: 
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_general_trade_signal(self,
                                     signal: SignalEvent,
                                     portfolio: AbstractPortfolio):
        """
        Handle an ambiguous ``TRADE`` :class:`SignalEvent`. It is up to the
        child classes to determine how to handle this. 
        
        :param signal: 
        :param portfolio: 
        :return: 
        """
        raise NotImplementedError

    def _handle_exit_signal(self,
                            portfolio: AbstractPortfolio,
                            signal: SignalEvent):
        """
        Handle an ``EXIT`` :class:`SignalEvent`.
        
        :param portfolio: 
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
                    'Owned qty is 0 for ticker: {signal.ticker}.')

        portfolio.blotter.place_order(
                signal.ticker, action, signal.order_type, qty,
                signal.stop_price, signal.limit_price)

    def _handle_cancel_signal(self,
                              portfolio: AbstractPortfolio,
                              signal: SignalEvent):
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
                order_type=signal.order_type)

    def _handle_hold_signal(self,
                            portfolio: AbstractPortfolio,
                            signal: SignalEvent):
        """
        Handle a ``HOLD`` :class:`SignalEvent`
        
        :param portfolio: 
        :param signal: 
        :return: 
        """
        portfolio.blotter.hold_all_orders_for_asset(signal.ticker)


class ClassicalBalancer(AbstractBalancer):
    """Balance based on Markowitz portfolio optimization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def balance(self, portfolio):
        pass


class AlwaysBalancedBalancer(AbstractBalancer):
    """Portfolio weights are always equal."""

    def __init__(self, include_cash=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_cash = include_cash

    def __call__(self,
                 portfolio: AbstractPortfolio,
                 signal: SignalEvent,
                 *args, **kwargs):
        """
        Balance the portfolio. 
        
        :param portfolio: 
        :param signal:
        :param args: 
        :param kwargs: 
        :return: 
        """
        current_weights = self._get_current_asset_weights(portfolio)

    def _get_target_qty(self,
                        portfolio: AbstractPortfolio,
                        signal: SignalEvent) -> int:
        """Return the target share quantity for a given event."""
        target_pct = 1 / len(portfolio.owned_assets)
        if self.include_cash:
            total = portfolio.total_value
        else:
            total = portfolio.total_asset_mv

        latest_adj_close = portfolio.bars.get_latest_bar_value(
                signal.ticker, pd_utils.ADJ_CLOSE_COL)
        target_qty = int(math.floor((target_pct * total) / latest_adj_close))

        self.logger.debug(f'target_qty: {target_qty}')

        return target_qty

    def _get_current_asset_weights(self,
                                   portfolio: AbstractPortfolio) \
            -> Dict[str, float]:
        weights = {}
        if self.include_cash:
            total_mv = portfolio.total_value
            weights['cash'] = portfolio.cash / total_mv
        else:
            total_mv = portfolio.total_asset_mv

        for ticker, asset in portfolio.owned_assets.items():
            weights[ticker] = asset.total_position_value / total_mv

        return weights

    def balance(self, portfolio):
        pass

import logging
import math
from abc import ABCMeta, abstractmethod
from typing import Dict

import pytech.utils.pandas_utils as pd_utils
from pytech.backtest.event import SignalEvent
from pytech.fin.portfolio import AbstractPortfolio


class AbstractBalancer(metaclass=ABCMeta):
    """Base class for all balancers."""

    def __init__(self,
                 portfolio: AbstractPortfolio,
                 allow_market_orders=True,
                 price_col=pd_utils.ADJ_CLOSE_COL,
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
        self.price_col = price_col
        self.portfolio = portfolio
        self.blotter = self.portfolio.blotter
        self.bars = self.portfolio.bars

    @abstractmethod
    def __call__(self,
                 signal: SignalEvent,
                 *args, **kwargs):
        """
        Balance the portfolio based on whatever methodology chosen.
        
        :param args: 
        :param kwargs: 
        :return: 
        """

    @abstractmethod
    def balance(self):
        """
        Balance the portfolio based on whatever methodology chosen.
        
        :return: 
        """
        raise NotImplementedError('Must implement balance(portfolio)')


class AlwaysBalancedBalancer(AbstractBalancer):
    """Portfolio weights are always equal."""

    def __init__(self,
                 portfolio: AbstractPortfolio,
                 allow_market_orders: bool = True,
                 price_col: str = pd_utils.ADJ_CLOSE_COL,
                 include_cash=False,
                 cash_reserves: float = .1,
                 *args, **kwargs):
        super().__init__(portfolio, allow_market_orders, price_col,
                         *args, **kwargs)
        self.include_cash = include_cash
        self.cash_reserves = cash_reserves

    def balance(self):
        if self.include_cash:
            total_mv = self.portfolio.total_value
        else:
            total_mv = self.portfolio.total_asset_mv

        weights = self._get_current_weights(self.portfolio)

        # add 1 because we will be buying another asset.
        target_pct = 1 / (len(self.portfolio.owned_assets) + 1)

        target_shares = {}

        for ticker in self.portfolio.owned_assets.keys():
            target_qty = self._get_target_qty(ticker, target_pct, total_mv)

            current_qty = self.portfolio.owned_assets[ticker].shares_owned

            diff = target_qty - current_qty

            # target_qty, target_mv, target_pct = self._get_targets(portfolio,
            #                                                       signal.ticker)

    def _get_target_qty(self,
                        ticker: str,
                        target_pct: float,
                        total_mv: float):
        price = self.bars.get_latest_bar_value(ticker, self.price_col)

        return int(math.floor((target_pct * total_mv) / price))

    def _get_targets(self,
                     portfolio: AbstractPortfolio,
                     ticker: str) -> (int, float, float):
        """Return the target share quantity for a given event."""
        target_pct = 1 / (len(portfolio.owned_assets) + 1)

        if self.include_cash:
            total_mv = portfolio.total_value
        else:
            total_mv = portfolio.total_asset_mv

        latest_adj_close = portfolio.bars.get_latest_bar_value(
                ticker, pd_utils.ADJ_CLOSE_COL)

        target_qty = int(
            math.floor((target_pct * total_mv) / latest_adj_close))

        target_mv = target_qty * latest_adj_close

        self.logger.debug(f'target_qty: {target_qty}')

        return target_qty, target_mv, target_pct

    def _get_current_weights(self,
                             portfolio: AbstractPortfolio) -> Dict[str, float]:
        weights = {}
        if self.include_cash:
            total_mv = portfolio.total_value
            weights['cash'] = portfolio.cash / total_mv
        else:
            total_mv = portfolio.total_asset_mv

        for ticker, asset in portfolio.owned_assets.items():
            weights[ticker] = asset.total_position_value / total_mv

        return weights

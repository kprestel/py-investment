import logging
import math
import cvxpy as cvx
import pytech.utils.pandas_utils as pd_utils
from abc import ABCMeta, abstractmethod
from pytech.fin.portfolio import AbstractPortfolio
from pytech.backtest.event import TradeSignalEvent


class AbstractBalancer(metaclass=ABCMeta):
    """Base class for all balancers."""

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Balance the portfolio based on whatever methodology chosen.
        
        :param args: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('Must implement __call__.')

    @abstractmethod
    def balance(self, portfolio: AbstractPortfolio):
        """
        Balance the portfolio based on whatever methodology chosen.
        
        :param AbstractPortfolio portfolio: 
        :return: 
        """
        raise NotImplementedError('Must implement balance(portfolio)')


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

    def __call__(self, portfolio, signal, *args, **kwargs):
        """
        Balance the portfolio. 
        
        :param AbstractPortfolio portfolio: 
        :param TradeSignalEvent signal:
        :param args: 
        :param kwargs: 
        :return: 
        """
        current_weights = self._get_current_asset_weights(portfolio)

    def _get_target_qty(self, portfolio: AbstractPortfolio,
                        signal: TradeSignalEvent):
        target_pct = 1 / len(portfolio.owned_assets)
        if self.include_cash:
            total = portfolio.total_value
        else:
            total = portfolio.total_asset_mv

        latest_adj_close = portfolio.bars.get_latest_bar_value(
                signal.ticker, pd_utils.ADJ_CLOSE_COL)
        return int(math.floor((target_pct * total) / latest_adj_close))

    def _get_current_asset_weights(self, portfolio: AbstractPortfolio):
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




import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

import pytech.data.reader as reader
import pytech.utils.pandas_utils as pd_utils
from utils.common_utils import tail


class EfficientFrontier(object):
    """Calculate the optimal portfolio based on the Efficient Frontier"""
    tickers: List[str]
    prices: List[float]
    rf: float

    def __init__(self, tickers: List[str] = None, rf: float = None):
        self.logger = logging.getLogger(__name__)
        if tickers is None:
            self.tickers = []
        else:
            self.tickers = tickers
        self.prices = self._load_data()
        # TODO make this a property and set it if its None
        self.rf = rf

    @property
    def rf(self):
        return self._rf

    @rf.setter
    def rf(self, val):
        if val is None:
            # TODO update this to be dynamic AF.
            self._rf = .015
        else:
            self._rf = val

    def __call__(self):
        expected_returns, covars = self._returns_covar()
        # self.logger.debug(f'returns: {expected_returns},\n'
        #                   f'covars: \n{covars}')
        return self._optimize_frontier(expected_returns, covars)

    def _load_data(self) -> List[float]:
        """Loads the data and updates `tickers` if needed."""
        if not self.tickers:
            for x in reader.get_symbols():
                self.tickers.append(x)

        tmp_prices_out = []
        max_prices = None

        # noinspection PyTypeChecker
        for t in self.tickers:
            df = reader.get_data(t, columns=[pd_utils.CLOSE_COL])
            prices = list(df[pd_utils.CLOSE_COL])
            num_prices = len(prices)
            if max_prices is None or num_prices < max_prices:
                # all must have the same amount of prices.
                max_prices = num_prices
            tmp_prices_out.append(prices)

        prices_out = []

        for l in tmp_prices_out:
            prices_out.append(list(tail(max_prices, l)))

        return prices_out

    def _returns_covar(self) -> Tuple[np.array, np.array]:
        """
        Calculate expected returns and covariance between assets.

        :return:
        """
        prices = np.matrix(self.prices)
        rows, cols = prices.shape
        returns = np.empty([rows, cols - 1])
        for r in range(rows):
            for c in range(cols - 1):
                p0, p1 = prices[r, c], prices[r, c + 1]
                returns[r, c] = (p1 / p0) - 1

        expected_returns = np.array([])
        for r in range(rows):
            expected_returns = np.append(expected_returns, np.mean(returns[r]))

        covars = np.cov(returns)
        # annualize returns and covars
        expected_returns = (1 + expected_returns) ** 252 - 1
        covars = covars * 252
        return expected_returns, covars

    def _optimize_frontier(self, returns, covar):
        weights = self._solve_weights(returns, covar)
        tan_mean, tan_var = _mean_var(weights, returns, covar)
        front_mean, front_var = self._solve_frontier(returns, covar)
        return _FrontierResult(self.tickers, weights, tan_mean, tan_var,
                               front_mean, front_var, covar, returns)

    def _solve_frontier(self, returns: np.array,
                        covar: np.array) -> Tuple[np.ndarray, np.ndarray]:
        def fitness(weights, returns, covar, r):
            mean, var = _mean_var(weights, returns, covar)
            penalty = 100 * abs(mean - r)
            return var + penalty

        frontier_mean = []
        frontier_var = []
        assets = len(returns)

        for r in np.linspace(np.min(returns), np.max(returns), num=20):
            # equal weights to start
            weights = np.ones([assets]) / assets
            # no shorts.
            b_ = [(0, 1) for _ in range(assets)]
            c_ = ({'type': 'eq', 'fun': lambda weights: sum(weights) - 1.0})
            optimized = minimize(fitness,
                                 weights,
                                 (returns, covar, r),
                                 method='SLSQP',
                                 constraints=c_,
                                 bounds=b_)
            if not optimized.success:
                raise BaseException(optimized.message)
            frontier_mean.append(r)
            frontier_var.append(_var(optimized.x, covar))

        return np.array(frontier_mean), np.array(frontier_var)

    def _solve_weights(self, returns: np.array,
                       covar: np.matrix) -> OptimizeResult:
        """
        Solve for the optimal weights.

        :param returns: numpy array of the average historical returns.
        :param covar: matrix of covariances.
        :return: optimal weights.
        """

        def fitness(weights, returns, covar, rf):
            mean, var = _mean_var(weights, returns, covar)
            sharpe = (mean - rf) / np.sqrt(var)
            return 1 / sharpe

        assets = len(returns)
        base_weights = np.ones([assets]) / assets
        # weights can be positive or negative
        b_ = [(0, 1) for _ in range(assets)]
        c_ = ({'type': 'eq', 'fun': lambda weights: sum(weights) - 1.0})
        optimized = minimize(fitness,
                             base_weights,
                             (returns, covar, self.rf),
                             method='SLSQP',
                             constraints=c_,
                             bounds=b_)
        if not optimized.success:
            raise BaseException(optimized.message)
        else:
            return optimized.x


class _FrontierResult(object):
    """Holds the results of the Frontier calculations."""

    def __init__(self, tickers, weights, tan_mean, tan_var, front_mean,
                 front_var, covar, returns):
        self.tickers = tickers
        self.weights = weights
        self.tan_mean = tan_mean
        self.tan_var = tan_var
        self.front_mean = front_mean
        self.front_var = front_var
        self.covar = covar
        self.returns = returns

    def __str__(self):
        df = pd.DataFrame({'Weight': self.weights}, index=self.tickers)
        c = pd.DataFrame(self.covar, columns=self.tickers, index=self.tickers)
        return (f'Weights: \n{df.T}\n'
                f'Covariances: \n{c}')

    def plot(self, asset_color: str = 'black',
             frontier_color: str = 'black',
             frontier_label: str = 'Frontier'):
        plt.style.use('ggplot')
        plt.scatter([self.covar[i, i] ** .5
                     for i in range(len(self.tickers))],
                    self.returns,
                    marker='x')

        for n in range(len(self.tickers)):
            plt.text(self.covar[n, n] ** .5,
                     self.returns[n],
                     f'  {self.tickers[n]}',
                     verticalalignment='center')

        plt.text(self.tan_var ** .5,
                 self.tan_mean,
                 '   tangent',
                 verticalalignment='center')
        plt.scatter(self.tan_var ** .5,
                    self.tan_mean)
        plt.plot(self.front_var ** .5,
                 self.front_mean,
                 label=frontier_label)
        plt.grid(True)
        plt.xlabel('variance')
        plt.ylabel('mean')
        plt.show()


def _mean(weights, returns):
    return sum(returns * weights)


def _var(weights, covar):
    return np.dot(np.dot(weights, covar), weights)


def _mean_var(weights, returns, covar):
    return _mean(weights, returns), _var(weights, covar)


if __name__ == '__main__':
    f = EfficientFrontier()
    r = f()

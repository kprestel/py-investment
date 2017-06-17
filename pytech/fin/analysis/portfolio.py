import numpy as np
from collections import Iterable
from scipy.optimize import minimize
import pytech.data.reader as reader
import pytech.utils.pandas_utils as pd_utils



def _mean(weights, returns):
    return sum(returns * weights)


def _var(weights, covar):
    return np.dot(np.dot(weights, covar), weights)


def _mean_var(weights, returns, covar):
    return _mean(weights, returns), _var(weights, covar)


def solve_frontier(returns: np.array,
                   covar: np.array,
                   rf: float):
    def fitness(weights, returns, covar, rf):
        mean, var = _mean_var(weights, returns, covar)
        penalty = 100 * abs(mean - rf)
        return var + penalty

    frontier_mean = []
    frontier_var = []
    assets = len(returns)

    for r in np.linespace(returns.min, returns.max):
        # equal weights
        weights = np.ones([assets]) / assets
        b_ = [(0, 1) for _ in range(assets)]
        c_ = ({'type': 'eq', 'func': lambda weights: sum(weights) - 1.0})
        optimized = minimize(fitness, weights, (returns, covar, r),
                             method='SLSQP',
                             constraints=c_, bounds=b_)
        if not optimized.success:
            raise BaseException(optimized.message)
        frontier_mean.append(r)
        frontier_var.append(_var(optimized.x, covar))

    return np.array(frontier_mean), np.array(frontier_var)


def solve_weights(returns: np.array,
                  covar: np.array,
                  rf: float):
    def fitness(weights, returns, covar, rf):
        mean, var = _mean_var(weights, returns, covar)
        util = (mean - rf) / np.sqrt(var)
        return 1 / util

    assets = len(returns)
    base_weights = np.ones([assets]) / assets
    b_ = [(0, 1) for _ in range(assets)]
    c_ = ({'type': 'eq', 'func': lambda weights: sum(weights) - 1.0})
    optimized = minimize(fitness, base_weights, (returns, covar, rf),
                         method='SLSQP',
                         constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    else:
        return optimized.x


def load_data(tickers:Iterable=None):
    if tickers is None:
        tickers = reader.get_symbols()

    prices_out = []

    for t in tickers:
        df = reader.get_data(t)
        prices = list(df[pd_utils.CLOSE_COL])
        prices_out.append(prices)

    return prices_out


import numpy as np
import pandas as pd
import pymc3 as pm


def monte_carlo(mu: float, vol: float, days: int, start_price: float,
                paths: int = 1000) -> np.ndarray:
    """
    Do a monte carlo sim.

    :param mu: The expected return.
    :param vol: The expected volatility.
    :param days: The number of trading days.
    :param start_price: The latest available price.
    :param paths: The number of paths to run for the simulation.
    :return: the expected value.
    """
    result = np.array([])
    avg_daily_return = mu / days
    daily_vol = vol / np.sqrt(days)

    for _ in range(paths):
        sim_returns = np.random.logistic(avg_daily_return,
                                         daily_vol,
                                         days) + 1

        price_list = np.array([start_price])

        # noinspection PyTypeChecker
        for x in sim_returns:
            price_list = np.append(price_list, price_list[-1] * x)

        result = np.append(result, price_list)

    return np.mean(result)


# noinspection PyTypeChecker
def _vol_model(df: pd.DataFrame):
    with pm.Model() as model:
        nu = pm.Exponential('nu', 1. / 10, testval=5.)
        sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
        s = pm.GaussianRandomWalk('s', sigma ** -2, shape=len(df.index))
        vol_process = pm.Deterministic('vol_process', pm.math.exp(-2 * s))
        r = pm.StudentT('r', nu, lam=1 / vol_process,
                        observed=df)
    with model:
        trace = pm.sample(20000)
    return trace


def mcmc(s: pd.Series):
    pass

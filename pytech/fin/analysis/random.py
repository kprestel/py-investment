import numpy as np


def monte_carlo(mu: float, vol: float, days: int,
                start_price: float, paths: int = 1000):
    """
    Do a monte carlo sim.

    :param mu: The expected return.
    :param vol: The expected volatility.
    :param days: The number of trading days.
    :param start_price: The latest available price.
    :param paths: The number of paths to run for the simulation.
    :return:
    """
    result = np.array([])
    for i in range(paths):
        sim_returns = np.random.normal(mu / days, vol / np.sqrt(days),
                                       days) + 1

        price_list = np.array([start_price])

        # noinspection PyTypeChecker
        for x in sim_returns:
            price_list = np.append(price_list, price_list[-1] * x)

        result = np.append(result, price_list)

    return np.mean(result)

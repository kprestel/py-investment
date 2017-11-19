import datetime as dt

from pytech.algo.strategy import CrossOverStrategy
from pytech.backtest.backtest import Backtest


class TestBacktest(object):

    def test_backtest_constructor(self, ticker_list, date_range):

        initial_capital = 100000
        # start_date = '2016-03-10'
        backtest = Backtest(ticker_list=['FB'],
                            date_range=date_range,
                            initial_capital=initial_capital,
                            strategy=CrossOverStrategy)

        assert isinstance(backtest, Backtest)
        backtest.run()



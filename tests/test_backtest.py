import datetime as dt

from pytech.algo.strategy import CrossOverStrategy
from pytech.backtest.backtest import Backtest


class TestBacktest(object):

    def test_backtest_constructor(self, ticker_list):

        initial_capital = 100000
        # start_date = '2016-03-10'
        start_date = dt.datetime(year=2016, month=3, day=10)
        end_date = dt.datetime(year=2016, month=5, day=10)
        backtest = Backtest(ticker_list=ticker_list,
                            initial_capital=initial_capital,
                            start_date=start_date,
                            end_date=end_date,
                            strategy=CrossOverStrategy)

        assert isinstance(backtest, Backtest)
        backtest.run()



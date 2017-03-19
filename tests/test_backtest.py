import pytest
from pytech.backtest.backtest import Backtest
from pytech.algo.strategy import BuyAndHold


class TestBacktest(object):

    def test_backtest_constructor(self, ticker_list):

        initial_capital = 100000
        start_date = '2016-03-10'
        backtest = Backtest(ticker_list=ticker_list, initial_capital=initial_capital, start_date=start_date,
                            strategy=BuyAndHold)

        assert isinstance(backtest, Backtest)
        backtest._run()



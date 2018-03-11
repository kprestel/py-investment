import pytest

from pytech.algo.strategy import CrossOverStrategy
from pytech.backtest.backtest import Backtest
from pytech.utils import DateRange


class TestBacktest(object):

    @pytest.mark.skip
    def test_backtest_constructor(self, ticker_list):
        date_range = DateRange('2017-06-01', '2017-07-01')

        initial_capital = 100000
        # start_date = '2016-03-10'
        backtest = Backtest(ticker_list=ticker_list,
                            date_range=date_range,
                            initial_capital=initial_capital,
                            strategy=CrossOverStrategy)

        assert isinstance(backtest, Backtest)
        backtest.run()



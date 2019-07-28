# noinspection PyUnresolvedReferences
import pytest
import pandas as pd

from pytech.data.reader import BarReader
from pytech.utils import DateRange


@pytest.fixture()
def date_range():
    """Need this to be shorter because of API limits."""
    return DateRange('2018-01-01', '2018-01-09')

class TestBarReader(object):
    reader = BarReader()

    @pytest.mark.vcr
    @pytest.mark.parametrize('ticker', ['FB', 'MSFT'])
    def test_get_data(self, date_range, ticker):
        min_dt = pd.Timestamp('2017-12-29 20:56:00+00:00')
        max_dt = pd.Timestamp('2018-01-10 01:36:00+00:00')
        # shape = (3172, 13)
        test = self.reader.get_data(ticker, date_range=date_range, freq='20min')
        assert min_dt == test.index.min()
        assert max_dt == test.index.max()
        # assert shape == test.shape
        for k, v in test.items():
            assert k is not None
            assert v is not None

    def test_get_symbols(self):
        # TODO make this more deterministic.
        for s in self.reader.tickers:
            assert s is not None

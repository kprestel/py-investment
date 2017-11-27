# noinspection PyUnresolvedReferences
import pytest

from pytech.data.reader import BarReader


class TestBarReader(object):
    reader = BarReader()

    def test_get_data(self, date_range):
        test = self.reader.get_data('GOOG', date_range=date_range)
        for k, v in test.items():
            assert k is not None
            assert v is not None

    def test_get_symbols(self):
        # TODO make this more deterministic.
        for s in self.reader.tickers:
            assert s is not None

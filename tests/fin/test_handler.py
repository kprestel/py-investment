import pytest

from pytech.fin.handler import AbstractSignalHandler


class TestSignalHandler(object):

    def test_get_correlation(self, basic_signal_handler: AbstractSignalHandler):
        df = basic_signal_handler.get_correlation_df()
        assert len(df.columns) == len(df.index)

import pytest
import pandas as pd
from pytech.data.handler import DataHandler, YahooDataHandler


class TestDataHandler(object):
    """Test the base abstract class"""

    def test_constructor(self):
        """Should always raise a TypeError because it is an abstract class."""
        with pytest.raises(TypeError):
            DataHandler()


class TestYahooDataHandler(object):
    """Test the :class:`YahooDataHandler`"""

    def test_constructor(self, events, ticker_list, start_date, end_date):
        """Test the constructor"""
        handler = YahooDataHandler(events, ticker_list, start_date, end_date)
        assert handler is not None

        for t in ticker_list:
            gen_df = handler.ticker_data[t]
            df = handler._get_new_bar('AAPL')
            # df = next(gen_df)[1]
            assert isinstance(df, pd.Series)


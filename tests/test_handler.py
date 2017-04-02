import pytest
from pytest import approx
import pandas as pd
import pytech.utils.pandas_utils as pd_utils
import pytech.utils.dt_utils as dt_utils
from pytech.data.handler import DataHandler, YahooDataHandler


# noinspection PyTypeChecker
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
            df = next(df)[1]
            # df = next(gen_df)[1]
            assert isinstance(df, pd.Series)

    def test_get_latest_bar_value(self, yahoo_data_handler):
        """
        Test getting the latest bar values.
        
        This test tests two days (a day is when update_bars() is called)
        and asserts that the correct values are present for the tickers.
        
        :param YahooDataHandler yahoo_data_handler: 
        """
        assert yahoo_data_handler is not None
        # yahoo_data_handler.update_bars()
        aapl_adj_close = (yahoo_data_handler
                          .get_latest_bar_value('AAPL',
                                                pd_utils.ADJ_CLOSE_COL))
        aapl_adj_close_expected = 99.07551600000001
        assert aapl_adj_close == approx(aapl_adj_close_expected)
        aapl_open = (yahoo_data_handler
                     .get_latest_bar_value('AAPL', pd_utils.OPEN_COL))
        aapl_open_expected = 101.410004
        assert aapl_open == approx(aapl_open_expected)

        fb_adj_close = (yahoo_data_handler
                        .get_latest_bar_value('FB', pd_utils.ADJ_CLOSE_COL))
        fb_adj_close_expected = 107.32
        assert fb_adj_close == approx(fb_adj_close_expected)

        fb_open = (yahoo_data_handler
                   .get_latest_bar_value('FB', pd_utils.OPEN_COL))
        fb_open_expected = 107.910004
        assert fb_open == approx(fb_open_expected)

        yahoo_data_handler.update_bars()

        aapl_adj_close = (yahoo_data_handler
                          .get_latest_bar_value('AAPL',
                                                pd_utils.ADJ_CLOSE_COL))
        aapl_adj_close_expected = 100.142954
        assert aapl_adj_close == approx(aapl_adj_close_expected)

        fb_adj_close = (yahoo_data_handler
                        .get_latest_bar_value('FB', pd_utils.ADJ_CLOSE_COL))
        fb_adj_close_expected = 109.410004
        assert fb_adj_close == approx(fb_adj_close_expected)

        with pytest.raises(KeyError):
            yahoo_data_handler.get_latest_bar_value('FAKE', pd_utils.OPEN_COL)

    def test_get_latest_bar_dt(self, yahoo_data_handler):
        """
        Test that the latest date returned is correct.
        
        :param YahooDataHandler yahoo_data_handler: 
        """
        test_date = yahoo_data_handler.get_latest_bar_dt('AAPL')
        assert test_date == dt_utils.parse_date('2016-03-10')

        yahoo_data_handler.update_bars()

        test_date = yahoo_data_handler.get_latest_bar_dt('AAPL')
        assert test_date == dt_utils.parse_date('2016-03-11')

        yahoo_data_handler.update_bars()

        test_date = yahoo_data_handler.get_latest_bar_dt('AAPL')
        assert test_date == dt_utils.parse_date('2016-03-14')

    def test_get_latest_bar(self, yahoo_data_handler):
        """
        Test getting the latest bar.
        
        :param YahooDataHandler yahoo_data_handler: 
        :return: 
        """
        bar = yahoo_data_handler.get_latest_bar('AAPL')
        dt = bar[pd_utils.DATE_COL]
        adj_close = bar[pd_utils.ADJ_CLOSE_COL]
        aapl_adj_close_expected = 99.07551600000001
        assert dt == '2016-03-10'
        assert adj_close == approx(aapl_adj_close_expected)
        yahoo_data_handler.update_bars()
        bar = yahoo_data_handler.get_latest_bar('AAPL')
        dt = bar[pd_utils.DATE_COL]
        assert dt == '2016-03-11'




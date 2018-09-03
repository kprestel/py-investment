import pytest
import pytz
from pytest import approx
import pandas as pd
import pytech.utils.pdutils as pd_utils
import pytech.utils.dt_utils as dt_utils
from pytech.data.handler import DataHandler, Bars


# noinspection PyTypeChecker
class TestDataHandler(object):
    """Test the base abstract class"""

    def test_constructor(self, events, ticker_list, date_range):
        """Test the constructor"""
        handler = Bars(events, ticker_list, date_range)
        assert handler is not None
        handler.update_bars()

        for t in ticker_list:
            df = handler.latest_ticker_data[t][0]
            assert isinstance(df, pd.Series)

    def test_get_latest_bar_value(self, bars):
        """
        Test getting the latest bar values.

        This test tests two days (a day is when update_bars() is called)
        and asserts that the correct values are present for the tickers.

        :param Bars bars:
        """
        assert bars is not None
        # yahoo_data_handler.update_bars()
        aapl_close = bars.latest_bar_value('AAPL', pd_utils.CLOSE_COL)
        aapl_close_expected = 143.93
        assert approx(aapl_close_expected) == aapl_close
        aapl_open = bars.latest_bar_value('AAPL', pd_utils.OPEN_COL)
        aapl_open_expected = 143.87
        assert approx(aapl_open_expected) == aapl_open

        fb_close = bars.latest_bar_value('FB', pd_utils.CLOSE_COL)
        fb_close_expected = 142.28
        assert fb_close == approx(fb_close_expected)

        fb_open = (bars.latest_bar_value('FB', pd_utils.OPEN_COL))
        fb_open_expected = 141.93
        assert fb_open == approx(fb_open_expected)

        bars.update_bars()

        aapl_close = bars.latest_bar_value('AAPL', pd_utils.CLOSE_COL)
        aapl_close_expected = 143.93
        assert aapl_close == approx(aapl_close_expected)

        fb_close = bars.latest_bar_value('FB', pd_utils.CLOSE_COL)
        fb_close_expected = 141.73
        assert fb_close == approx(fb_close_expected)

        with pytest.raises(KeyError):
            bars.latest_bar_value('FAKE', pd_utils.OPEN_COL)

    def test_get_latest_bar_dt(self, bars):
        """
        Test that the latest date returned is correct.

        :param Bars bars:
        """
        test_date = bars.get_latest_bar_dt('AAPL')
        expected = dt_utils.parse_date('2017-06-20 15:44:00', tz=pytz.UTC)
        assert expected == test_date

        bars.update_bars()

        test_date = bars.get_latest_bar_dt('AAPL')
        expected = dt_utils.parse_date('2017-06-20 15:45:00', tz=pytz.UTC)
        assert expected == test_date

        bars.update_bars()

        test_date = bars.get_latest_bar_dt('AAPL')
        expected = dt_utils.parse_date('2017-06-20 15:46:00', tz=pytz.UTC)
        assert expected == test_date

    def test_get_latest_bar(self, bars):
        """
        Test getting the latest bar.

        :param Bars bars:
        :return:
        """
        bar = bars.get_latest_bar('AAPL')
        dt = dt_utils.parse_date(bar.name, tz=pytz.UTC)
        close = bar[pd_utils.CLOSE_COL]
        aapl_close_expected = 145.21
        expected_dt = dt_utils.parse_date('2017-06-20 15:44:00', tz=pytz.UTC)
        assert expected_dt == dt
        assert close == approx(aapl_close_expected)
        bars.update_bars()
        bar = bars.get_latest_bar('AAPL')
        dt = dt_utils.parse_date(bar.name, tz=pytz.UTC)
        expected_dt = dt_utils.parse_date('2017-06-20 15:45:00', tz=pytz.UTC)
        assert expected_dt == dt

    def test_make_agg_df(self, bars: Bars):
        """Test creating the agg df"""
        df = bars.make_agg_df()

        if 'SPY' not in bars.tickers:
            # it is expected for there to be 1 more column if the market
            # ticker isn't in the data_handler
            assert len(df.columns) == len(bars.tickers) + 1
        else:
            assert len(df.columns) == len(bars.tickers)


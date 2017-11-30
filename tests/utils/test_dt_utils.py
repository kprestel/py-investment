import datetime as dt

import pytest
import pandas_market_calendars as mcal
import pytz

import pytech.utils as utils


@pytest.mark.parametrize('start,end,start_expected,end_expected', [
    (None, None, dt.datetime(2010, 1, 1),
     utils.prev_weekday(dt.datetime.today())),
    (2011, 2016, dt.datetime(2011, 1, 1), dt.datetime(2016, 1, 1)),
    (dt.datetime(2010, 6, 1), dt.datetime(2017, 1, 1),
     dt.datetime(2010, 6, 1), dt.datetime(2017, 1, 1))
])
def test_sanitize_dates(start, end, start_expected, end_expected):
    """Check date() to avoid issues with seconds"""
    start, end = utils.sanitize_dates(start, end)
    assert start.date() == start_expected.date()
    assert end.date() == end_expected.date()


@pytest.mark.parametrize('adate,expected', [
    (dt.datetime(2017, 1, 1), False),
    (dt.datetime(2017, 1, 18), True)
])
def test_is_weekday(adate, expected):
    assert utils.is_trade_day(adate) == expected


@pytest.mark.parametrize('adate,expected', [
    (dt.datetime(2017, 1, 18), dt.datetime(2017, 1, 18)),
    (dt.datetime(2017, 5, 7), dt.datetime(2017, 5, 5))
])
def test_prev_weekday(adate, expected):
    assert utils.prev_weekday(adate) == expected

@pytest.mark.parametrize('dt,expected,exchange', [
    (dt.datetime(2017, 6, 1), dt.datetime(2017, 6, 1, 16), 'NYSE'),
    (dt.datetime(2017, 5, 1), dt.datetime(2017, 5, 1, 16), 'NYSE')
])
def test_parse_date(dt, expected, exchange):
    cal = mcal.get_calendar(exchange)
    expected = expected.replace(tzinfo=cal.tz)
    expected = expected.astimezone(pytz.UTC)
    test_dt = utils.parse_date(date_to_parse=dt, exchange=exchange)
    assert test_dt == expected

class TestDateRange(object):

    def test_dt_index(self):
        date_range = utils.DateRange()
        assert date_range.dt_index is not None

import datetime as dt

import pytest
import pytz
import pandas as pd
import pandas_market_calendars as mcal
from hypothesis import (
    given,
    reproduce_failure,
)
from hypothesis.extra.numpy import datetime64_dtypes
from hypothesis.extra.pytz import timezones
from hypothesis.strategies import (
    datetimes,
    data,
    sampled_from,
    one_of,
    none,
    dates,
)

import pytech.utils as utils

aware_dts = one_of(datetimes(max_value=dt.datetime.now(),
                             min_value=dt.datetime(2010, 1, 1),
                             timezones=one_of(timezones(), none())),
                   dates(max_value=dt.date.today(), min_value=dt.date.today()))


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


@pytest.mark.parametrize('dt,expected,exchange,tz', [
    (dt.datetime(2017, 6, 1), dt.datetime(2017, 6, 1, 16, tzinfo=pytz.UTC),
     'NYSE', pytz.UTC),
    (dt.datetime(2017, 5, 1), dt.datetime(2017, 5, 1, 16, tzinfo=pytz.UTC),
     'NYSE', pytz.UTC),
    (pd.to_datetime(dt.datetime(2017, 6, 1, 20, 56, tzinfo=pytz.UTC)),
     dt.datetime(2017, 6, 1, 20, 56, tzinfo=pytz.UTC), 'NYSE', pytz.UTC)
])
def test_parse_date_(dt, expected, exchange, tz):
    test_dt = utils.parse_date(date_to_parse=dt, exchange=exchange, tz=tz)
    assert test_dt == expected


@given(aware_dts, timezones(), data(),
       sampled_from(('close_time', 'open_time')))
def test_parse_date(dt_, tz1, data, default_time):
    cal = data.draw(sampled_from(('NYSE', 'CME', 'ICE')))
    try:
        tz1 = dt_.tzinfo
    except AttributeError:
        tz1 = None
    test = utils.parse_date(dt_, exchange=cal, tz=tz1,
                            default_time=default_time)
    cal = mcal.get_calendar(cal)
    exchange_time = getattr(cal, default_time)
    if isinstance(dt_, dt.date) and not isinstance(dt_, dt.datetime):
        dt_ = dt.datetime(dt_.year, dt_.month, dt_.day, exchange_time.hour,
                          exchange_time.minute, exchange_time.second,
                          exchange_time.microsecond, tzinfo=cal.tz)
        dt_ = dt_.astimezone(pytz.UTC)
    elif isinstance(dt_, dt.datetime):
        dt_ = utils.replace_time(dt_, exchange_time)
        if dt_.tzinfo is None and tz1 is None:
            dt_ = dt_.replace(tzinfo=cal.tz)
        elif dt_.tzinfo is None:
            dt_ = dt_.replace(tzinfo=tz1)
        dt_ = dt_.astimezone(pytz.UTC)
    assert test == dt_


class TestDateRange(object):

    def test_dt_index(self):
        date_range = utils.DateRange()
        assert date_range.dt_index is not None

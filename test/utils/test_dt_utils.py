import datetime as dt
from typing import Union

import pandas as pd
import pandas_market_calendars as mcal
import pytest
import pytz
from hypothesis import (
    assume,
    given,
)
from hypothesis.strategies import (
    dates,
    datetimes,
    one_of,
    sampled_from,
)

import pytech.utils as utils

dts = one_of(datetimes(max_value=dt.datetime.now(),
                       min_value=dt.datetime(2010, 1, 1)),
             dates(max_value=dt.date.today(), min_value=dt.date(2010, 1, 1)))

tzs = sampled_from((pytz.UTC, pytz.timezone('America/New_York'),
                   pytz.timezone('America/Chicago'),
                   pytz.timezone('US/Eastern'),
                   None)
                   )

@given(dts, dts, tzs, sampled_from(('NYSE', 'CME', 'ICE')),
       sampled_from(('close_time', 'open_time')))
def test_sanitize_dates(start, end, tz, cal, default_time):
    assume(type(start) == type(end))
    assume(start < end)
    if isinstance(start, dt.datetime):
        start = start.replace(tzinfo=tz)
        end = end.replace(tzinfo=tz)

    test_start, test_end = utils.sanitize_dates(start, end,
                                                exchange=cal,
                                                tz=tz,
                                                default_time=default_time)

    start = _clean_expected_dt(start, tz, cal, default_time)
    end = _clean_expected_dt(end, tz, cal, default_time)

    if not utils.is_trade_day(start, exchange=cal):
        start = utils.prev_trade_day(start, exchange=cal)

    if not utils.is_trade_day(end, exchange=cal):
        end = utils.prev_trade_day(end, exchange=cal)

    if start >= end:
        start = _clean_expected_dt(start, tz, cal, 'open_time', force=True)
        if not utils.is_trade_day(start, exchange=cal):
            start = utils.prev_trade_day(start, exchange=cal)

        end = _clean_expected_dt(end, tz, cal, 'close_time', force=True)
        if not utils.is_trade_day(end, exchange=cal):
            end = utils.prev_trade_day(end, exchange=cal)

    assert start == test_start
    assert end == test_end


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
    assert utils.prev_trade_day(adate, 'NYSE') == expected


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


def _clean_expected_dt(dt_: Union[dt.datetime, dt.date],
                       tz1: pytz.timezone,
                       cal: str,
                       default_time: str,
                       force: bool = False):
    cal = mcal.get_calendar(cal)
    exchange_time = getattr(cal, default_time)
    if exchange_time.tzinfo is None:
        exchange_time = exchange_time.replace(tzinfo=cal.tz)
    if isinstance(dt_, dt.date) and not isinstance(dt_, dt.datetime):
        dt_ = dt.datetime(dt_.year, dt_.month, dt_.day, exchange_time.hour,
                          exchange_time.minute, exchange_time.second,
                          exchange_time.microsecond, tzinfo=cal.tz)
        dt_ = dt_.astimezone(pytz.UTC)
    elif isinstance(dt_, dt.datetime):
        dt_ = utils.replace_time(dt_, exchange_time, force)
        # dt_ = dt_.replace(tzinfo=cal.tz).astimezone(pytz.UTC)
        if dt_.tzinfo is None and tz1 is None:
            dt_ = dt_.replace(tzinfo=cal.tz)
        elif dt_.tzinfo is None:
            dt_ = dt_.replace(tzinfo=tz1)
        dt_ = dt_.astimezone(pytz.UTC)
    return dt_


@given(dts, sampled_from(('NYSE', 'CME', 'ICE')),
       sampled_from(('close_time', 'open_time')))
def test_parse_date(dt_, cal, default_time):
    try:
        tz1 = dt_.tzinfo
    except AttributeError:
        tz1 = None
    test = utils.parse_date(dt_, exchange=cal, tz=tz1,
                            default_time=default_time)
    dt_ = _clean_expected_dt(dt_, tz1, cal, default_time)
    assert test == dt_


class TestDateRange(object):

    def test_dt_index(self):
        date_range = utils.DateRange()
        assert date_range.dt_index is not None

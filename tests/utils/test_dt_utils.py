import datetime as dt

import pytest

import pytech.utils.dt_utils as dt_utils


@pytest.mark.parametrize('start,end,start_expected,end_expected', [
    (None, None, dt.datetime(2010, 1, 1),
     dt_utils.prev_weekday(dt.datetime.today())),
    (2011, 2016, dt.datetime(2011, 1, 1), dt.datetime(2016, 1, 1)),
    (dt.datetime(2010, 6, 1), dt.datetime(2017, 1, 1),
     dt.datetime(2010, 6, 1), dt.datetime(2017, 1, 1))
])
def test_sanitize_dates(start, end, start_expected, end_expected):
    """Check date() to avoid issues with seconds"""
    start, end = dt_utils.sanitize_dates(start, end)
    assert start.date() == start_expected.date()
    assert end.date() == end_expected.date()


@pytest.mark.parametrize('adate,expected', [
    (dt.datetime(2017, 1, 1), False),
    (dt.datetime(2017, 1, 18), True)
])
def test_is_weekday(adate, expected):
    assert dt_utils.is_weekday(adate) == expected


@pytest.mark.parametrize('adate,expected', [
    (dt.datetime(2017, 1, 18), dt.datetime(2017, 1, 18)),
    (dt.datetime(2017, 5, 7), dt.datetime(2017, 5, 5))
])
def test_prev_weekday(adate, expected):
    assert dt_utils.prev_weekday(adate) == expected

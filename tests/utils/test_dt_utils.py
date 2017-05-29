import datetime as dt

import pytest

import pytech.utils.dt_utils as dt_utils


@pytest.mark.parametrize('start,end,start_expected,end_expected', [
    (None, None,
     dt.datetime(2010, 1, 1), dt.datetime.today()),
    (2011, 2016,
     dt.datetime(2011, 1, 1), dt.datetime(2016, 1, 1)),
    (dt.datetime(2010, 6, 1), dt.datetime(2017, 1, 1),
     dt.datetime(2010, 6, 1), dt.datetime(2017, 1, 1))
])
def test_sanitize_dates(start, end, start_expected, end_expected):
    """Check date() to avoid issues with seconds"""
    start, end = dt_utils.sanitize_dates(start, end)
    assert start.date() == start_expected.date()
    assert end.date() == end_expected.date()

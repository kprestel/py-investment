import datetime as dt
from typing import (
    Tuple,
    Union,
)

import pandas as pd
import pandas_market_calendars as mcal
import pytz
from pandas.api.types import is_number

from pytech.exceptions import (
    PyInvestmentTypeError,
    PyInvestmentValueError,
)

date_type = Union[dt.datetime, pd.Timestamp]

NYSE = mcal.get_calendar('NYSE')


def parse_date(date_to_parse: date_type):
    """
    Converts strings or datetime objects to UTC timestamps.

    :param date_to_parse: The date to parse.
    :type date_to_parse: datetime or str or Timestamp
    :return: ``pandas.TimeStamp``
    """
    if isinstance(date_to_parse, dt.date) and not isinstance(date_to_parse,
                                                             dt.datetime):
        raise PyInvestmentTypeError(f'date must be a datetime object. '
                                    f'{type(date_to_parse)} was provided')
    elif isinstance(date_to_parse, pd.Timestamp):
        if date_to_parse.tz is None:
            return date_to_parse.replace(tzinfo=pytz.UTC)
        else:
            return date_to_parse
    elif isinstance(date_to_parse, dt.datetime):
        return pd.to_datetime(date_to_parse.replace(tzinfo=pytz.UTC),
                              utc=True)
    elif isinstance(date_to_parse, dt.date):
        return pd.to_datetime(date_to_parse, utc=True)
    elif isinstance(date_to_parse, str):
        # TODO: timezone
        return pd.to_datetime(date_to_parse, utc=True)
    else:
        raise PyInvestmentTypeError(
            'date_to_parse must be a pandas '
            'Timestamp, datetime, or a date string. '
            f'{type(date_to_parse)} was provided')


def get_default_date(is_start_date):
    if is_start_date:
        temp_date = dt.datetime.now() - dt.timedelta(days=365)
        return parse_date(temp_date)
    else:
        return parse_date(dt.datetime.now())


def sanitize_dates(start, end) -> Tuple[dt.datetime, dt.datetime]:
    """
    Return a tuple of (start, end)

    if start is `None` then default is 1/1/2010
    if end is `None` then default is today.
    """
    if is_number(start):
        # treat ints as a year
        start = dt.datetime(start, 1, 1)
    start = pd.to_datetime(start, utc=True)

    if is_number(end):
        end = dt.datetime(end, 1, 1)
    end = pd.to_datetime(end, utc=True)

    if start is None:
        start = dt.datetime(2010, 1, 1, tzinfo=pytz.UTC)

    if end is None:
        end = dt.datetime.today()
        end = end.replace(tzinfo=pytz.UTC)
        end = prev_weekday(end)

    # return parse_date(start), parse_date(end)
    return start, end


def is_trade_day(a_dt: date_type):
    """
    True if `dt` is a weekday.

    Monday = 1
    Sunday = 7
    """
    return a_dt.isoweekday() < 6 and a_dt.date() not in NYSE.holidays().holidays


def prev_weekday(a_dt: date_type):
    """
    Returns last weekday from a given date.

    If the given date is a weekday it will be returned.
    """
    if is_trade_day(a_dt):
        return a_dt

    while a_dt.isoweekday() > 5:
        a_dt -= dt.timedelta(days=1)
    return a_dt


class DateRange(object):
    def __init__(self, start: date_type = None,
                 end: date_type = None,
                 freq: str = 'B',
                 cal: str = 'NYSE'):
        # TODO: open, closed
        self.start, self.end = sanitize_dates(start, end)
        self.freq = freq
        if cal == 'NYSE':
            self.cal = NYSE
        else:
            raise NotImplementedError('TODO.')

        if self.start > self.end:
            raise PyInvestmentValueError(f'start must be less than end.'
                                         f'start: {start}, end: {end}')

    def is_trade_day(self, dt: str):
        if dt == 'start':
            return is_trade_day(self.start)
        elif dt == 'end':
            return is_trade_day(self.end)
        else:
            raise PyInvestmentValueError(f'{dt} is not valid. Must be '
                                         f'"start" or "end"')

    @property
    def dt_index(self):
        schedule = self.cal.schedule(start_date=self.start, end_date=self.end)
        return mcal.date_range(schedule=schedule, frequency=self.freq)

    def __repr__(self):
        return (f'{self.__class__.__name__}(start={self.start}, '
                f'end={self.end}, freq={self.freq}, cal={self.cal})')

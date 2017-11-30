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
    PyInvestmentKeyError,
)

date_type = Union[dt.datetime, pd.Timestamp, str]

NYSE = mcal.get_calendar('NYSE')


def parse_date(date_to_parse: date_type,
               exchange: str = 'NYSE') -> dt.datetime:
    """
    Converts strings or datetime objects to UTC timestamps.

    :param date_to_parse: the date to parse.
    :param exchange: the exchange being used, defaults to NYSE.
        This is used to set the time in the event that the `date_date_to_parse`
        does not have a time set.
    :return: ``pandas.TimeStamp``
    """
    from dateutil.parser import parse
    try:
        cal = mcal.get_calendar(exchange)
    except KeyError:
        raise PyInvestmentKeyError(f'{exchange} is not a valid exchange.')

    if isinstance(date_to_parse, str):
        try:
            date_to_parse = parse(date_to_parse)
        except ValueError as e:
            raise PyInvestmentValueError(e)
        except TypeError as ex:
            raise PyInvestmentTypeError(ex)

    exchange_time = cal.close_time

    repl = {}
    for attr in ("hour", "minute", "second", "microsecond"):
        value = getattr(date_to_parse, attr, False)
        if value:
            repl[attr] = value
        else:
            repl[attr] = getattr(exchange_time, attr)

    date_to_parse = date_to_parse.replace(**repl)

    try:
        if date_to_parse.tz is None:
            date_to_parse = date_to_parse.replace(tzinfo=cal.tz)
    except AttributeError:
        if date_to_parse.tzinfo is None:
            date_to_parse = date_to_parse.replace(tzinfo=cal.tz)

    return date_to_parse.astimezone(pytz.UTC)


def get_default_date(is_start_date):
    if is_start_date:
        temp_date = dt.datetime.now(pytz.UTC) - dt.timedelta(days=365)
        return parse_date(temp_date)
    else:
        return parse_date(dt.datetime.now(pytz.UTC))


def sanitize_dates(start, end) -> Tuple[dt.datetime, dt.datetime]:
    """
    Return a tuple of (start, end)

    if start is `None` then default is 1/1/2010
    if end is `None` then default is today.
    """
    if is_number(start):
        # treat ints as a year
        start = dt.datetime(start, 1, 1)
    start = parse_date(start)

    if is_number(end):
        end = dt.datetime(end, 1, 1)
    end = parse_date(end)

    if start is None:
        start = dt.datetime(2010, 1, 1)

    if end is None:
        end = dt.datetime.now(pytz.UTC)
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
                 freq: str = '1D',
                 cal: str = 'NYSE'):
        # TODO: open, closed
        self.start, self.end = sanitize_dates(start, end)
        self.freq = freq
        if cal == 'NYSE':
            self.cal = NYSE
        else:
            raise NotImplementedError('TODO.')

        if self.start > self.end:
            raise PyInvestmentValueError(f'start must be less than end. '
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

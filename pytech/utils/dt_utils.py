import datetime as dt
from typing import (
    Tuple,
    Union,
)

import pandas as pd
import pandas_market_calendars as mcal
import pytz
from pandas_market_calendars import MarketCalendar

from pytech.exceptions import DateParsingError
from . import digits

date_type = Union[dt.datetime, pd.Timestamp, str, int]

NYSE = mcal.get_calendar('NYSE')


def parse_date(date_to_parse: date_type,
               exchange: str = 'NYSE',
               tz: pytz.timezone = None,
               default_time: str = 'close_time') -> dt.datetime:
    """
    Converts strings or datetime objects to UTC timestamps.

    :param date_to_parse: the date to parse.
    :param exchange: the exchange being used, defaults to NYSE.
        This is used to set the time in the event that the `date_date_to_parse`
        does not have a time set.
    :param tz:
    :param default_time: The time to use in the event that the `date_to_parse`
        does not have a `time`.
        * close_time
        * open_time

    :return: ``dt.datetime``
    """
    from dateutil.parser import parse
    if date_to_parse is None:
        raise DateParsingError('None is not a valid date')

    if isinstance(date_to_parse, pd.Timestamp):
        date_to_parse = date_to_parse.to_pydatetime()

    try:
        cal = mcal.get_calendar(exchange)
    except KeyError:
        raise KeyError(f'{exchange} is not a valid exchange.')

    if isinstance(date_to_parse, str):
        try:
            date_to_parse = parse(date_to_parse)
        except (TypeError, AttributeError):
            raise DateParsingError(date=date_to_parse)

    exchange_time = getattr(cal, default_time)
    if exchange_time.tzinfo is None:
        exchange_time = exchange_time.replace(tzinfo=cal.tz)
    date_to_parse = replace_time(date_to_parse, exchange_time)

    if date_to_parse.tzinfo is None and tz is None:
        date_to_parse = date_to_parse.replace(tzinfo=cal.tz)
    elif date_to_parse.tzinfo is None and tz is not None:
        date_to_parse = date_to_parse.replace(tzinfo=tz)

    return date_to_parse.astimezone(pytz.UTC)


def replace_time(dt_: Union[dt.date, dt.datetime],
                 time_: dt.time,
                 force: bool = False) -> dt.datetime:
    """
    Takes a time object and replaces the `time` of the `dt_` if is it not set

    :param dt_:
    :param time_:
    :return:
    """
    try:
        repl = {}
        for attr in ('hour', 'minute', 'second', 'microsecond', 'tzinfo'):
            value = getattr(dt_, attr, False)
            if value and not force:
                repl[attr] = value
            else:
                repl[attr] = getattr(time_, attr)

        if isinstance(dt_, dt.date):
            dt_ = dt.datetime(dt_.year,
                              dt_.month,
                              dt_.day)

        dt_ = dt_.replace(**repl)
        return dt_
    except AttributeError as e:
        raise DateParsingError(date=dt_) from e


def get_default_date(is_start_date):
    if is_start_date:
        temp_date = dt.datetime.now(pytz.UTC) - dt.timedelta(days=365)
        return parse_date(temp_date)
    else:
        return parse_date(dt.datetime.now(pytz.UTC))


def sanitize_dates(start: date_type,
                   end: date_type,
                   exchange: Union[str, MarketCalendar] = 'NYSE',
                   tz: pytz.timezone = None,
                   default_time: str = 'close_time') -> Tuple[
    dt.datetime, dt.datetime]:
    """
    Return a tuple of (start, end)

    if start is `None` then default is 1/1/2010
    if end is `None` then default is today.
    """

    def _parse_date(dt_):
        return parse_date(dt_, exchange=exchange,
                          tz=tz, default_time=default_time)

    def int_to_dt(n):
        digits_ = digits(n)
        if digits_ == 4:
            # treat ints as a year
            return _parse_date(dt.datetime(n, 1, 1))

        else:
            return _parse_date(n)

    try:
        start = _parse_date(start)
    except DateParsingError as e:
        if start is None:
            start = dt.datetime(2010, 1, 1)
        elif isinstance(start, int):
            start = int_to_dt(start)
        else:
            raise e
        start = _parse_date(start)

    if not is_trade_day(start, exchange=exchange):
        start = prev_trade_day(start, exchange)

    try:
        end = _parse_date(end)
    except DateParsingError as e:
        if end is None:
            end = dt.datetime.now(pytz.UTC)
        elif isinstance(end, int):
            end = int_to_dt(end)
        else:
            raise e
        end = _parse_date(end)

    if not is_trade_day(end, exchange=exchange):
        end = prev_trade_day(end, exchange)

    if start >= end:
        if start.date() == end.date():
            if not isinstance(exchange, MarketCalendar):
                exchange = mcal.get_calendar(exchange)

            open_time = getattr(exchange, 'open_time')
            close_time = getattr(exchange, 'close_time')

            start = replace_time(start, open_time, force=True)
            end = replace_time(end, close_time, force=True)

            start = start.replace(tzinfo=exchange.tz).astimezone(pytz.UTC)

            if not is_trade_day(start, exchange):
                start = prev_trade_day(start, exchange=exchange)

            end = end.replace(tzinfo=exchange.tz).astimezone(pytz.UTC)

            if not is_trade_day(end, exchange):
                end = prev_trade_day(end, exchange=exchange)
        else:
            raise ValueError(f'start date: {start} cannot be greater than '
                             f'or equal to end date: {end}')

    return start, end


def is_trade_day(a_dt: date_type, exchange: Union[MarketCalendar, str] = NYSE):
    """
    True if `dt` is a weekday.

    Monday = 1
    Sunday = 7
    """
    try:
        holidays = exchange.holidays().holidays
    except AttributeError:
        holidays = mcal.get_calendar(exchange).holidays().holidays

    try:
        dt_ = a_dt.date()
    except AttributeError:
        dt_ = a_dt

    return a_dt.isoweekday() < 6 and dt_ not in holidays


def prev_trade_day(a_dt: date_type, exchange):
    """
    Returns last weekday from a given date.

    If the given date is a weekday it will be returned.
    """
    if is_trade_day(a_dt, exchange=exchange):
        return a_dt

    while not is_trade_day(a_dt, exchange=exchange):
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
            raise ValueError(f'start must be less than end. '
                             f'start: {start}, end: {end}')

    def is_trade_day(self, dt: str):
        if dt == 'start':
            return is_trade_day(self.start, self.cal)
        elif dt == 'end':
            return is_trade_day(self.end, self.cal)
        else:
            raise ValueError(f'{dt} is not valid. Must be "start" or "end"')

    @property
    def dt_index(self):
        schedule = self.cal.schedule(start_date=self.start, end_date=self.end)
        return mcal.date_range(schedule=schedule, frequency=self.freq)

    def __repr__(self):
        return (f'{self.__class__.__name__}(start={self.start}, '
                f'end={self.end}, freq={self.freq}, cal={self.cal})')

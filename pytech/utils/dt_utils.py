import datetime as dt
from typing import Tuple, Union

import pandas as pd
from dateutil import tz
from pandas.core.dtypes.inference import is_number
from pandas.tslib import Timestamp

from pytech.utils.decorators import memoize


@memoize
def parse_date(date_to_parse: Union[dt.datetime, Timestamp]):
    """
    Converts strings or datetime objects to UTC timestamps.

    :param date_to_parse: The date to parse.
    :type date_to_parse: datetime or str or Timestamp
    :return: ``pandas.TimeStamp``
    """
    if isinstance(date_to_parse, dt.date) and not isinstance(date_to_parse,
                                                             dt.datetime):
        raise TypeError(
                'date must be a datetime object. {} was provided'.format(
                        type(date_to_parse)))
    elif isinstance(date_to_parse, Timestamp):
        if date_to_parse.tz is None:
            return date_to_parse.tz_localize('UTC')
        else:
            return date_to_parse
    elif isinstance(date_to_parse, dt.datetime):
        return pd.to_datetime(date_to_parse.replace(tzinfo=tz.tzutc()),
                              utc=True)
    elif isinstance(date_to_parse, dt.date):
        return pd.to_datetime(date_to_parse, utc=True)
    elif isinstance(date_to_parse, str):
        # TODO: timezone
        return pd.to_datetime(date_to_parse, utc=True)
    else:
        raise TypeError(
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
    start = pd.to_datetime(start)

    if is_number(end):
        end = dt.datetime(end, 1, 1)
    end = pd.to_datetime(end)

    if start is None:
        start = dt.datetime(2010, 1, 1)

    if end is None:
        end = dt.datetime.today()

    return start, end

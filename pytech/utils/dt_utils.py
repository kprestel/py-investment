from datetime import date, datetime, timedelta

import pandas as pd
from dateutil import tz


def parse_date(date_to_parse):
    """
    Converts strings or datetime objects to UTC timestamps.

    :param date_to_parse: The date to parse.
    :type date_to_parse: datetime or str or Timestamp
    :return: ``pandas.TimeStamp``
    """
    if isinstance(date_to_parse, date) and not isinstance(date_to_parse,
                                                          datetime):
        raise TypeError(
                'date must be a datetime object. {} was provided'.format(
                        type(date_to_parse)))
    elif isinstance(date_to_parse, pd.TimeSeries):
        if date_to_parse.tz is None:
            return date_to_parse.tz_localize('UTC')
        else:
            return date_to_parse
    elif isinstance(date_to_parse, datetime):
        return pd.to_datetime(date_to_parse.replace(tzinfo=tz.tzutc()),
                              utc=True)
    elif isinstance(date_to_parse, date):
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
        temp_date = datetime.now() - timedelta(days=365)
        return parse_date(temp_date)
    else:
        return parse_date(datetime.now())

from dateutil import parser
from datetime import date, datetime


def parse_date(date_to_parse):
    try:
        return parser.parse(date_to_parse)
    except ValueError:
        raise ValueError('Unable to parse {} to a date_to_parse, must be a string formatted as a date_to_parse '
                         'or a datetime obj'.format(date_to_parse))
    except TypeError:
        if type(date_to_parse) == date:
            raise TypeError('date must be a datetime object')
        else:
            return date_to_parse
    except AttributeError:
        return date_to_parse


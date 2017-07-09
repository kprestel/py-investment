import datetime as dt
import itertools

import pytech.utils as utils
from pytech.data.reader import BarReader


class Market(object):
    """
    Singleton like object that is used to represent market data such as
    the SPY and various interest rates.
    """
    _shared_state = None

    def __init__(self, ticker: str = 'SPY',
                 source: str = 'google',
                 start_date: dt.datetime = None,
                 end_date: dt.datetime = None,
                 lib_name: str = 'pytech.market'):
        if self._shared_state is None:
            self._shared_state = self.__dict__
            start_date, end_date = utils.dt_utils.sanitize_dates(start_date,
                                                                 end_date)
            self.ticker = ticker
            self.start_date = start_date
            self.end_date = end_date
            self.lib_name = lib_name
            self.reader = BarReader(lib_name)
            self.source = source
            self.market = self.reader.get_data(self.ticker,
                                               self.source,
                                               self.start_date,
                                               self.end_date)
        else:
            self.__dict__ = self._shared_state


class BondBasket(object):
    US_TBONDS = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS5', 'DGS7',
                 'DGS10', 'DGS20', 'DGS30']
    US_TBILLS = ['TB3MS', 'DTB6', 'DTB4WK', 'TB3MS', 'TB6MS', 'DTB1YR']
    LIBOR = ['USDONTD156N', 'USD1MTD156N', 'USD1WKD156N',
             'USD3MTD156N', 'USD6MTD156N', 'USD12MTD156N']
    ALL = itertools.chain(US_TBONDS, US_TBILLS, LIBOR)
    SOURCE = 'fred'
    _shared_state = None

    def __init__(self, start_date: dt.datetime = None,
                 end_date: dt.datetime = None):
        if self._shared_state is None:
            self._shared_state = self.__dict__
            start_date, end_date = utils.dt_utils.sanitize_dates(start_date,
                                                                 end_date)
            self.start_date = start_date
            self.end_date = end_date


class YieldCurve(object):
    def __init__(self):
        pass

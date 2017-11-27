import itertools

import pytech.utils as utils
from pytech.data.reader import BarReader
from pytech.utils import DateRange


class Market(utils.Borg):
    """
    Singleton like object that is used to represent market data such as
    the SPY and various interest rates.
    """

    def __init__(self, ticker: str = 'SPY',
                 source: str = 'google',
                 date_range: DateRange = None,
                 lib_name: str = 'pytech.market') -> None:
        super().__init__()
        self.date_range = date_range or DateRange()
        self.ticker = ticker
        self.lib_name = lib_name
        self.reader = BarReader()
        self.source = source
        self.market = self.reader.get_data(self.ticker,
                                           self.source,
                                           self.date_range)


class BondBasket(utils.Borg):
    US_TBONDS = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS5', 'DGS7',
                 'DGS10', 'DGS20', 'DGS30']
    US_TBILLS = ['TB3MS', 'DTB6', 'DTB4WK', 'TB3MS', 'TB6MS', 'DTB1YR']
    LIBOR = ['USDONTD156N', 'USD1MTD156N', 'USD1WKD156N',
             'USD3MTD156N', 'USD6MTD156N', 'USD12MTD156N']
    ALL = itertools.chain(US_TBONDS, US_TBILLS, LIBOR)
    SOURCE = 'fred'

    def __init__(self, date_range: DateRange = None) -> None:
        super().__init__()
        self.date_range = date_range or DateRange()
        self.reader = BarReader()
        self.data = self.reader.get_data(self.ALL, self.SOURCE,
                                         self.date_range)


class YieldCurve(object):
    def __init__(self):
        pass

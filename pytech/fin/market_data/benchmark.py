import pytech.utils.dt_utils as dt_utils
import pandas_datareader.data as web

class BenchMark(object):

    def __init__(self, start_date=None, end_date=None, ticker='^GPSC'):

        if start_date is None:
            self.start_date = dt_utils.get_default_date(is_start_date=True)
        else:
            self.start_date = dt_utils.parse_date(start_date)

        if end_date is None:
            self.end_date = dt_utils.get_default_date(is_start_date=False)
        else:
            self.end_date = dt_utils.parse_date(end_date)

        self.ticker = ticker

        self.ohlcv = web.DataReader(ticker, 'yahoo', start=self.start_date, end=self.end_date)



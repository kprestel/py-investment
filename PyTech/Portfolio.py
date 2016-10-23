import pandas as pd
import pandas_datareader.data as web
import datetime

from PyTech.Stock import Stock
from PyTech import analysis


class Portfolio:
    def __init__(self, tickers, start=None, end=None, bench='^GSPC'):
        if type(tickers) != list:
            # make sure tickers is a list
            tickers = [tickers]
        # ensure start and end are proper type
        if start is None:
            # default to 1 year
            self.start = datetime.datetime.today() - datetime.timedelta(days=365)
        elif type(start) != datetime.datetime:
            raise TypeError('start must be a datetime.datetime')
        else:
            self.start = start

        if end is None:
            # default to day
            self.end = datetime.datetime.today()
        elif type(end) != datetime.datetime:
            raise TypeError('end must be a datetime.datetime')
        else:
            self.end = end
        self.asset_dict = {}
        self.benchmark = web.DataReader(bench, 'yahoo', start=self.start, end=self.end)

        for ticker in tickers:
            self.asset_dict[ticker] = Stock(ticker, self.start, self.end)


    def sma(self):
        for ticker, stock in self.asset_dict.items():
            yield stock.simple_moving_average()


if __name__ == "__main__":
    portfolio = Portfolio(tickers=['AAPL', 'SPY'])
    for i in portfolio.asset_dict.values():
        print(i.sma)
    for i in portfolio.sma():
        print(i.tail())

    # def simple_moving_average(self, period, column):















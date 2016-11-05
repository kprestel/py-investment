import pandas as pd
import pandas_datareader.data as web
import datetime

from PyTech.Stock import Stock
from xbrl import XBRLParser, GAAP, GAAPSerializer
from PyTech import analysis


class Portfolio:
    def __init__(self, tickers, start=None, end=None, bench='^GSPC', starting_cash=1000000):
        if type(tickers) != list:
            # make sure tickers is a list
            tickers = [tickers]
        # ensure start and end are proper type
        if start is None:
            # default to 1 year
            # self.start = datetime.datetime.today() - datetime.timedelta(days=365)
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
        self.cash = starting_cash

        for ticker in tickers:
            self.asset_dict[ticker] = Stock(ticker, self.start, self.end)

    def buy_shares(self, ticker, num_shares, buy_date):
        if ticker in self.asset_dict:
            pass

    def portfolio_return(self):
        pass

    def sma(self):
        for ticker, stock in self.asset_dict.items():
            yield stock.simple_moving_average()


if __name__ == "__main__":
    import requests
    import xml.etree.ElementTree as et
    from io import BytesIO, StringIO

    head = {'Content-Type': 'application/xml'}
    r = requests.get('https://www.sec.gov/Archives/edgar/data/320193/000162828016020309/aapl-20160924.xml', headers=head)
    r.encoding = 'XML'
    temp = BytesIO(r.content)
    # temp = StringIO(r.text)
    # temp = open(r.content)
    tree = et.parse(temp)
    # tree = et.parse(r.content)
    root = tree.getroot()
    print(root)

    # with open()
    x = XBRLParser()
    # y = x.parse(file_handle='https://www.sec.gov/Archives/edgar/data/320193/000162828016020309/aapl-20160924.xml')
    with open(temp, 'rb+') as f:
        y = x.parse(file_handle=f)
    # y = x.parse(file_handle=temp)
    gaap = x.parse(y, doc_date='20160924', context='current', ignore_errors=0)
    s = GAAPSerializer()
    rs = s.dump(gaap)
    print(rs.data)
    # portfolio = Portfolio(tickers=['AAPL', 'SPY', 'SKX'])
    # for i in portfolio.asset_dict.values():
    #     i.simple_median_crossover_signals()
        # print(i.sma_crossover_signals())
        # print(i.sma)
        # print(i.beta)
    # for i in portfolio.sma():
    #     print(i.tail())

        # def simple_moving_average(self, period, column):

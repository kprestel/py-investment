import pandas as pd
import pandas_datareader.data as web
import datetime
from crawler.spiders.edgar import EdgarSpider, URLGenerator

from pytech.stock import Stock
from pytech import analysis


class Portfolio(object):
    def __init__(self, tickers, start=None, end=None, bench='^GSPC', starting_cash=1000000):
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
            # default to today
            self.end = datetime.datetime.today()
        elif type(end) != datetime.datetime:
            raise TypeError('end must be a datetime.datetime')
        else:
            self.end = end
        self.asset_dict = {}
        # benchmark defaults to the S&P 500
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
    # testing stuff
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    # process = CrawlerProcess({'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'})
    process = CrawlerProcess(get_project_settings())
    # spider = EdgarSpider(symbols='AAPL', startdate='20160101', enddate='20161104')
    dict = {'symbols': 'AAPL', 'start_date':'20160101', 'end_date': '20161104'}
    process.crawl(EdgarSpider, **dict)
    process.start()

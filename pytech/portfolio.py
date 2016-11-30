import pandas as pd
import pandas_datareader.data as web
from datetime import date, timedelta
from dateutil import parser
from sqlalchemy import orm

from pytech.stock import Base, HasStock
from sqlalchemy import Column, Numeric, String, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declared_attr

class PortfolioAsset(object):
    """
    Mixin object to create a one to many relationship with Portfolio

    Any class that inherits from this class may be added to a Portfolio's asset_list which will place a foreign key
    in that asset class and create a relationship between it and the Portfolio that owns it
    """

    @declared_attr
    def portfolio_id(cls):
        return Column('ticker', ForeignKey('portfolio.id'))

    @declared_attr
    def portfolio(cls):
        return relationship('Portfolio')

class Portfolio(Base):
    """
    Holds stocks and keeps tracks of the owner's cash as well as makes Trades
    """

    # TODO: figure out how to make trades and what the relationship it should have with stocks
    """
    NOTES:
        What is the best way to model the Portfolio -> Stock -> Trade relationship?
            Use the Trade table as an association table?
        How else can price/qty be tracked for a specific Stock -> Portfolio?
        How else can we handle this?
    """

    id = Column(Integer, primary_key=True)
    cash = Column(Numeric(30, 2))
    benchmark = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)

    def __init__(self, tickers, start_date=None, end_date=None, benchmark='^GSPC', starting_cash=1000000):
        if type(tickers) != list:
            # make sure tickers is a list
            tickers = [tickers]

        # ensure start_date and end_date are proper type
        # I don't like the way this is done but don't have a better idea right now
        if start_date is None:
            # default to 1 year
            self.start = date.today() - timedelta(days=365)
        else:
            try:
                self.start = parser.parse(start_date).date()
            except ValueError:
                raise ValueError('Error parsing start_date to date. {} was provided')
            except TypeError:
                self.start = start_date.date()

        if end_date is None:
            # default to today
            self.end = date.today()
        else:
            try:
                self.end = parser.parse(end_date).date()
            except ValueError:
                raise ValueError('Error parsing end_date to date. {} was provided')
            except TypeError:
                self.end = end_date.date()

        self.asset_dict = {}
        # benchmark defaults to the S&P 500
        self.benchmark = web.DataReader(benchmark, 'yahoo', start=self.start, end=self.end)
        self.cash = starting_cash

    @orm.reconstructor
    def init_on_load(self):
        """
        :return:

        recreate the benchmark series on load from DB
        """
        # TODO: THIS
        pass

        # for ticker in tickers:
        #     self.asset_dict[ticker] = Stock(ticker, self.start, self.end)

    def buy_shares(self, ticker, num_shares, buy_date):
        if ticker in self.asset_dict:
            pass

    def portfolio_return(self):
        pass

    def sma(self):
        for ticker, stock in self.asset_dict.items():
            yield stock.simple_moving_average()

class Trade(HasStock, Base):
    id = Column(Integer, primary_key=True)
    trade_date = Column(DateTime)
    # 'buy' or 'sell'
    action = Column(String)
    # 'long' or 'short'
    position = Column(String)
    qty = Column(Integer)
    price_per_share = Column(Numeric(9,2))
    corresponding_trade_id = Column(Integer, ForeignKey('trade.id'))
    corresponding_trade = relationship('Trade', remote_side=[id])

    def __init__(self, trade_date, qty, price_per_share, stock, action='buy', position=None, corresponding_trade=None):
        """
        :param trade_date: datetime.datetime, corresponding to the trade date
        :param qty: int, number of shares traded
        :param price_per_share: float, price per individual share in the trade or the average share price in the trade
        :param stock: Stock, the stock object that was traded
        :param action: str, buy or sell depending on what kind of trade it was
        :param position: str, long or short
        """
        try:
            self.trade_date = parser.parse(trade_date).date()
        except ValueError:
            raise ValueError('Error parsing trade_date into a date. {} was provided'.format(trade_date))
        except TypeError:
            self.trade_date = trade_date.date()

        if action.lower() == 'buy' or action.lower() == 'sell':
            # TODO: may have to run a query to check if we own the stock or not? and if we do use update?
            self.action = action.lower()
        else:
            raise ValueError('action must be either "buy" or "sell". {} was provided.'.format(action))

        if position.lower() == 'long' or position.lower() == 'short':
            self.position = position.lower()
        elif position is None and corresponding_trade is not None:
            self.position = position
        elif position is None and corresponding_trade is None:
            raise ValueError('position can only be None if a corresponding_trade is also provided and None was provided')
        else:
            raise ValueError('position must be either "long" or "short". {} was provided.'.format(position))

        if isinstance(stock, Stock):
            self.stock = stock
        else:
            raise ValueError('stock must be an instance of the Stock class. {} was provided.'.format(stock))

        if corresponding_trade is None or isinstance(corresponding_trade, Trade):
            # TODO: check if the corresponding_trade is actually in the DB yet
            self.corresponding_trade = corresponding_trade
        else:
            raise ValueError('corresponding_trade must either be None or an instance of a Trade object')

        # TODO: if the position is short shouldn't this be negative?
        self.qty = qty
        self.price_per_share = price_per_share

if __name__ == "__main__":
    # testing stuff
    from scrapy.crawler import CrawlerProcess, Crawler
    from scrapy.utils.project import get_project_settings
    from twisted.internet import reactor
    from pytech import Session
    from pytech.stock import Stock
    tickers = ['AAPL', 'F', 'SKX']
    start = '20160101'
    end = '20161124'
    session = Session()
    stock = Stock(ticker='AAPL', start_date=start, end_date=end, get_fundamentals=True)
    session.add(stock)
    session.commit()
    # stock_dict = Stock.create_stock_fundamentals_from_list(ticker_list=tickers, start=start, end=end)
    # for k, v in stock_dict.items():
    #     print('key: {}'.format(k))
    #     print('val: {}'.format(v))
    # print(stock_dict)
    # process = CrawlerProcess({'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'})
    # process = CrawlerProcess(get_project_settings())
    # spider = EdgarSpider(symbols='AAPL', startdate='20160101', enddate='20161104')
    # dict = {'symbols': 'AAPL', 'start_date':'20160101', 'end_date': '20161104'}
    # dict_one = {'symbols': 'F', 'start_date':'20160101', 'end_date': '20161104'}
    # dict_two = {'symbols': 'SKX', 'start_date':'20160101', 'end_date': '20161104'}
    # to_crawl = [dict, dict_one, dict_two]
    # running = []
    # for d in to_crawl:
    #     settings = get_project_settings()
    #     crawler = Crawler(EdgarSpider, settings=settings)
    #     running.append(crawler)
        # crawler.signals.connect(spider)
        # crawler.configure()
        # crawler.crawl(EdgarSpider, settings, **d)
        # crawler.start()
    # reactor.run()
    # process.crawl(EdgarSpider, **dict)
    # process.crawl(EdgarSpider, **dict_one)
    # process.crawl(EdgarSpider, **dict_two)
    # process.start()
    # process.join()

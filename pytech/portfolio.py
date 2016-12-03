# from pytech import Session
import pandas as pd
from pytech.base import Base
import pandas_datareader.data as web
from datetime import date, timedelta, datetime
from dateutil import parser
from sqlalchemy import ForeignKey
from sqlalchemy import orm

import pytech.db_utils as db
from pytech.stock import HasStock, Stock, OwnedStock
from sqlalchemy import Column, Numeric, String, DateTime, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
import logging

logger = logging.getLogger(__name__)


class AssetUniverse(Base):

    watched_assets = relationship('Stock', backref='asset_universe',
                          collection_class=attribute_mapped_collection('ticker'),
                          cascade='all, delete-orphan')



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

        The portfolio class ideally is the only class that will interact with the database. By that I mean that no other
        class should be 'committing' anything the only way anything gets committed to the db is when the Portfolio they
        are all directly or indirectly associated with. I'm not 100% sure this will be possible or the best design
        pattern but it kinda seems like the right idea right now.  Except for the whole spider thing... so we will
        see where the future takes us.

        One thing I know for sure, its gonna be a bumpy ride.

    your's truly:
        KP.

    """

    cash = Column(Numeric(30, 2))
    benchmark_ticker = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    assets = relationship('OwnedStock', backref="portfolio",
                        collection_class=attribute_mapped_collection('ticker'),
                        cascade='all, delete-orphan')

    def __init__(self, tickers, start_date=None, end_date=None, benchmark_ticker='^GSPC', starting_cash=1000000,
                 get_fundamentals=False, get_ohlcv=True):
        """
        :param tickers:
            a list, containing all the tickers in the portfolio. This list will be used to create the Stock
            objects that they correspond to, there cannot be any duplicates
        :param start_date:
            a date, the start date of the analysis.
            This will be passed to each :class: Stock created and the ohlcv data frame loaded will start at this date.
            start_date will default to today - 365 days if nothing is passed in
        :param end_date:
            a date, the end date of the analysis.
            This will be passed in each :class: Stock created and the ohlcv data frame as well.
            end_date defaults to today
        :param benchmark_ticker:
            a string, the ticker of the market index or benchmark to compare the portfolio against.
            benchmark_ticker defaults to the S&P 500
        :param starting_cash:
            float, the amount of dollars to allocate to the portfolio initially
        :param get_fundamentals:
            a boolean, if True the fundamentals of each :class:`Stock` will be retrieved
            NOTE: if a lot of stocks are loaded this may take a little bit of time
            get_fundamentals defaults to False
        :param get_ohlcv:
            a boolean, if True an ohlcv data frame will be created for each :class:`Stock`
            get_ohlcv defaults to True
        """
        if type(tickers) != list:
            # make sure tickers is a list
            tickers = [tickers]

        # ensure start_date and end_date are proper type
        # I don't like the way this is done but don't have a better idea right now

        if start_date is None:
            # default to 1 year
            self.start_date = date.today() - timedelta(days=365)
        else:
            try:
                self.start_date = parser.parse(start_date).date()
            except ValueError:
                raise ValueError('Error parsing start_date to date. {} was provided')
            except TypeError:
                self.start_date = start_date.date()

        if end_date is None:
            # default to today
            self.end_date = date.today()
        else:
            try:
                self.end_date = parser.parse(end_date).date()
            except ValueError:
                raise ValueError('Error parsing end_date to date. {} was provided')
            except TypeError:
                self.end_date = end_date.date()

        self.assets = {}
        if get_fundamentals:
            stocks = Stock.from_ticker_list(ticker_list=tickers, start=start_date, end=end_date,
                                            get_ohlcv=get_ohlcv)
            for stock in stocks:
                self.assets[stock.ticker] = stock

        self.benchmark_ticker = benchmark_ticker

        self.benchmark = web.DataReader(benchmark_ticker, 'yahoo', start=self.start_date, end=self.end_date)
        self.cash = starting_cash

    @orm.reconstructor
    def init_on_load(self):
        """
        :return:

        recreate the benchmark series on load from DB
        """
        self.benchmark = web.DataReader(self.benchmark_ticker, 'yahoo', start=self.start_date, end=self.end_date)

    def buy_shares(self, ticker, num_shares, buy_date):
        if ticker in self.assets:
            pass

    def portfolio_return(self):
        pass

    def sma(self):
        for ticker, stock in self.assets.items():
            yield stock.simple_moving_average()

class Trade(HasStock, Base):
    """
    This class is used to make trades and keep trade of past trades
    """
    trade_date = Column(DateTime)
    action = Column(String)
    position = Column(String)
    qty = Column(Integer)
    price_per_share = Column(Numeric(9,2))
    corresponding_trade_id = Column(Integer, ForeignKey('trade.id'))
    # corresponding_trade = relationship('Trade', remote_side=[id])

    def __init__(self, trade_date, qty, average_price, stock, action='buy', position=None, corresponding_trade=None):
        """
        :param trade_date: datetime, corresponding to the date and time of the trade date
        :param qty: int, number of shares traded
        :param average_price: float
            price per individual share in the trade or the average share price in the trade
        :param stock:
            a :class: Stock, the stock object that was traded
        :param action: str, must be *buy* or *sell* depending on what kind of trade it was
        :param position: str, must be *long* or *short*
        """
        try:
            self.trade_date = parser.parse(trade_date)
        except ValueError:
            raise ValueError('Error parsing trade_date into a date. {} was provided.'.format(trade_date))
        except TypeError:
            if type(trade_date) == datetime:
                self.trade_date = trade_date
            else:
                raise TypeError('trade_date must be a datetime object or a string. '
                                '{} was provided'.format(type(trade_date)))

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
            raise ValueError('Nice try buy, position must be either "long" or "short". {} was provided.'.format(position))

        try:
            self.stock = stock.make_trade(qty=qty, average_price=average_price)
        except AttributeError:
            try:
                self.stock = OwnedStock(ticker=stock.ticker, shares_owned=qty, average_share_price=average_price,
                                   purchase_date=self.trade_date)
            except AttributeError:
                raise AttributeError('stock must be a Stock object. {} was provided'.format(type(stock)))

        if corresponding_trade is None or isinstance(corresponding_trade, Trade):
            # TODO: check if the corresponding_trade is actually in the DB yet
            self.corresponding_trade = corresponding_trade
        else:
            raise ValueError('corresponding_trade must either be None or an instance of a Trade object.'
                             '{} was provided'.format(type(corresponding_trade)))

        # TODO: if the position is short shouldn't this be negative? and does this really belong here?
        self.qty = qty
        self.price_per_share = average_price


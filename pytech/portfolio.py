# from pytech import Session
import pandas as pd
import pytech.utils as utils
from pytech.errors import AssetExistsException, AssetNotInUniverseException
from pytech.base import Base
import pandas_datareader.data as web
from datetime import date, timedelta, datetime
from dateutil import parser
from sqlalchemy import ForeignKey
from sqlalchemy import orm

import pytech.db_utils as db
from pytech.stock import HasStock, Stock, OwnedStock, OwnsStock
from sqlalchemy import Column, Numeric, String, DateTime, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
import logging

logger = logging.getLogger(__name__)


class AssetUniverse(Base):
    """
    This class will contain all the :class:`Asset` that are eligible to be bought and sold.  The intention is to allow
    the user to limit what stocks they are watching and willing to trade.  It allows for analysis to be done on each
    :class:`Asset` and compare them against all other possible trade opportunities a user is comfortable with making.
    """
    start_date = Column(DateTime)
    assets = relationship('Stock', backref='asset_universe',
                          collection_class=attribute_mapped_collection('ticker'),
                          cascade='all, delete-orphan')

    def __init__(self, ticker_list=None, start_date=None, end_date=None, get_fundamentals=False, get_ohlcv=True):
        """
        :param ticker_list: containing all the ticker_list in the portfolio. This list will be used to create the
            :class:``Stock`` objects that they correspond to, there cannot be any duplicates. This is only if you want
            to load stocks upon initialization, it can be updated later.
        :type ticker_list: list
        :param start_date: a date, the start date of the analysis. This will be passed to each :class:``Stock``
            created and the ohlcv data frame loaded will start at this date.
            (default: end_date - 365 days)
        :type start_date: datetime
        :param end_date: the end date of the analysis. This will be passed in each :class:``Stock`` created
            and the ohlcv data frame as well.
            (default: ``datetime.now()``)
        :type end_date: datetime
        :param get_fundamentals: if True the fundamentals of each :class:``Stock`` ticker will be retrieved
            NOTE: if a lot of stocks are loaded this may take a little bit of time, but without :class:`Fundamental`
             loaded for each :class:``Asset`` fundamental analysis will not work.
            (default: False)
        :type get_fundamentals: bool
        :param get_ohlcv: a boolean, if True an ohlcv data frame will be created for each :class:`Stock`
            (default: True)
        :type get_ohlcv: bool
        """
        if ticker_list:
            self.ticker_list = ticker_list
        else:
            self.ticker_list = []

        if start_date is None:
            self.start_watch_date = datetime.now() - timedelta(days=365)
        else:
            self.start_watch_date = utils.parse_date(start_date)
        if end_date is None:
            # default to today
            self.end_date = datetime.now()
        else:
            self.end_date = utils.parse_date(end_date)

        self.assets = {}
        if get_fundamentals:
            watched_stocks = Stock.from_ticker_list(ticker_list=ticker_list, start=self.start_watch_date,
                                                    end=self.end_date, get_ohlcv=get_ohlcv)
            for stock in watched_stocks:
                self.assets[stock.ticker] = stock

    def add_assets(self, ticker, start_date, end_date=None, get_fundamentals=True, get_ohlcv=True):
        if self.assets.get(ticker):
            raise AssetExistsException('Asset is already in the {} assets dict.'.format(self.__class__))
        start_date = utils.parse_date(start_date)
        if end_date is None:
            end_date = start_date - timedelta(days=365)
        else:
            end_date = utils.parse_date(end_date)
        self.assets[ticker] = Stock(ticker=ticker, start_date=start_date, end_date=end_date,
                                    get_fundamentals=get_fundamentals, get_ohlcv=get_ohlcv)
        with db.transactional_session as session:
            session.add(self)

    def add_assets_from_list(self, ticker_list, start_date, end_date=None, get_fundamentals=True, get_ohlcv=True):
        for ticker in ticker_list:
            if self.assets.get(ticker):
                raise AssetExistsException('Asset is already in the {} assets dict.'.format(self.__class__))
            self.assets[ticker] = Stock(ticker=ticker, start_date=start_date, end_date=end_date,
                                        get_fundamentals=get_fundamentals, get_ohlcv=get_ohlcv)



class Portfolio(AssetUniverse):
    """
    This class inherits from :class:``AssetUniverse`` so that it has access to its analysis functions.  It is important to
    note that :class:``Portfolio`` is its own table in the database as it represents the :class:``Asset`` that the user
    currently owns.  An :class:``Asset`` can be in both the *asset_universe* table as well as the *portfolio* table but
    a :class:``Asset`` does have to be in the :class:``AssetUniverse`` in order to be traded.

    Holds stocks and keeps tracks of the owner's cash as well and their :class:`OwnedAssets` and allows them to perform
    analysis on just the :class:`Asset` that they currently own.
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

        This idea has already changed.  See the db_utils.py file.

        One thing I know for sure, its gonna be a bumpy ride.

    your's truly:
        KP.

    """
    __tablename__ = 'portfolio'

    id = Column(Integer, primary_key=True)
    cash = Column(Numeric(30, 2))
    benchmark_ticker = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    assets = relationship('OwnedStock', backref='portfolio',
                        collection_class=attribute_mapped_collection('ticker'),
                        cascade='all, delete-orphan')
    __mapper_args__ = {
        'concrete': True
    }

    def __init__(self, ticker_list, start_date=None, end_date=None, benchmark_ticker='^GSPC', starting_cash=1000000,
                 get_fundamentals=True, get_ohlcv=True):
        """
        :param benchmark_ticker: the ticker of the market index or benchmark to compare the portfolio against.
            (default: *^GSPC*)
        :type benchmark_ticker: str
        :param starting_cash: the amount of dollars to allocate to the portfolio initially
            (default: 10000000)
        :type starting_cash: long
        """
        super().__init__(ticker_list=ticker_list, start_date=start_date, end_date=end_date,
                         get_fundamentals=get_fundamentals, get_ohlcv=get_ohlcv)

        # if get_fundamentals:
        #     stocks = OwnedStock.from_ticker_list(ticker_list=ticker_list, start=start_date, end=end_date,
        #                                          get_ohlcv=get_ohlcv)
        #     for stock in stocks:
        #         self.assets[stock.ticker] = stock

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


    # def _make_trade(self, ticker, qty, strategy, trade_date=None, action='buy', price_per_share=None):
    #     owned_stock = self.assets.get(ticker)
    #     if owned_stock:
    #         post_trade_stock = owned_stock.make_trade(qty=qty, price_per_share=price_per_share)
    #         if isinstance(post_trade_stock, OwnedStock):
    #             self.assets[ticker] = post_trade_stock
    #         else:
    #             del self.assets[ticker]
    #             # with db.transactional_session as session:
    #             #     session.add(Trade(qty=qty, price_per_share=post_trade_stock))
    #     else:
    #         with db.transactional_session as session:
    #             stock = session.query(Stock).filter(Stock.ticker == ticker).first()
    #             if stock:
    #                 owned_stock = OwnedStock(ticker=stock.ticker, shares_owned=qty)
    #                 self.assets[ticker] = owned_stock
    #                 session.add(Trade(qty=qty, price_per_share=owned_stock.average_share_price, stock=owned_stock,
    #                                   strategy=strategy, action=action, trade_date=trade_date))
    #                 session.add(self)
    #             else:
    #                 raise AssetNotInUniverseException('{} could not be located in the AssetUniverse so the trade was '
    #                                                   'aborted'.format(ticker))

    def make_trade(self, ticker, qty, action, price_per_share=None, trade_date=None):
        owned_asset = self.assets.get(ticker)
        if owned_asset:
            self._update_existing_position(qty=qty, action=action, price_per_share=price_per_share,
                                           trade_date=trade_date, asset=owned_asset)
        else:
            self._open_new_position(ticker=ticker, qty=qty, action=action, price_per_share=price_per_share,
                                    trade_date=trade_date)

    def _open_new_position(self, ticker, qty, price_per_share, trade_date, action):
        """

        Create a new :class:``OwnedStock`` object associated with this portfolio as well as update the cash position

        :param qty: how many shares are being bought or sold.
            If the position is a **long** position use a negative number to close it and positive to open it.
            If the position is a **short** position use a negative number to open it and positive to close it.
        :type qty: int
        :param price_per_share: the average price per share in the trade.
            This should always be positive no matter what the trade's position is.
        :type price_per_share: long
        :param trade_date: the date and time the trade takes place
            (default: now)
        :type trade_date: datetime
        :return: None

        This method processes the trade and then writes the results to the database. It will create a new instance of
        :class:``OwnedStock`` class and at it to the :class:``Portfolio`` asset dict.
        """
        asset = self._get_asset_from_universe(ticker=ticker)
        if asset:
            if action.lower() == 'sell':
                # if selling an asset that is not in the portfolio that means it has to be a short sale.
                position = 'short'
            else:
                position = 'long'
            owned_asset = OwnedStock(ticker=asset.ticker, shares_owned=qty, average_share_price=price_per_share,
                                     position=position)
            # inverse the total position's value and credit cash for that much
            self.cash += owned_asset.total_position_value * -1
            self.assets[owned_asset.ticker] = owned_asset
            trade = Trade(qty=qty, price_per_share=price_per_share, stock=owned_asset, action='buy',
                          strategy='Open new {} position'.format(position), trade_date=trade_date)
            with db.transactional_session as session:
                session.add(self)
                session.add(trade)
        else:
            raise AssetNotInUniverseException('{} could not be located in the AssetUniverse so the trade was '
                                              'aborted'.format(ticker))

    def _get_asset_from_universe(self, ticker):
        with db.query_session as session:
            return session.query(Stock).filter(Stock.ticker == ticker).first()

    def _update_existing_position(self, qty, price_per_share, trade_date, asset, action):
        """
        Update the :class:``OwnedStock`` associated with this portfolio as well as the cash position

        :param qty: how many shares are being bought or sold.
            If the position is a **long** position use a negative number to close it and positive to open it.
            If the position is a **short** position use a negative number to open it and positive to close it.
        :type qty: int
        :param price_per_share: the average price per share in the trade.
            This should always be positive no matter what the trade's position is.
        :type price_per_share: long
        :param trade_date: the date and time the trade takes place
            (default: now)
        :type trade_date: datetime
        :param asset: the asset that is already in the portfolio
        :type asset: OwnedStock
        :return: None

        This method processes the trade and then writes the results to the database.
        """
        if asset.total_position_value < 0:
            position = 'short'
        else:
            position = 'long'
        post_trade_asset = asset.make_trade(qty=qty, price_per_share=price_per_share)
        if post_trade_asset:
            self.assets[post_trade_asset.ticker] = post_trade_asset
            self.cash += post_trade_asset.total_position_value * -1
            trade = Trade(qty=qty, price_per_share=price_per_share, stock=post_trade_asset,
                          strategy='Update an existing {} position'.format(position), action=action,
                          trade_date=trade_date)
        else:
            self.cash += asset.total_position_value
            del self.assets[asset.ticker]
            # add this to the db?
            unowned_asset = Stock.from_dict(asset.__dict__)
            trade = Trade(qty=qty, price_per_share=price_per_share, stock=unowned_asset,
                          strategy='Close an existing {} position'.format(position), action=action,
                          trade_date=trade_date)
        with db.transactional_session as session:
            session.add(self)
            session.add(trade)

    def portfolio_return(self):
        pass

    def sma(self):
        for ticker, stock in self.assets.items():
            yield stock.simple_moving_average()

class Trade(OwnsStock, Base):
    """
    This class is used to make trades and keep trade of past trades
    """
    # id = Column(Integer, primary_key=True)
    trade_date = Column(DateTime)
    action = Column(String)
    strategy = Column(String)
    qty = Column(Integer)
    price_per_share = Column(Numeric)
    corresponding_trade_id = Column(Integer, ForeignKey('trade.id'))
    net_trade_value = Column(Numeric)
    # owned_stock_id = Column(Integer, ForeignKey('owned_stock.id'))
    # owned_stock = relationship('OwnedStock')
    # corresponding_trade = relationship('Trade', remote_side=[id])

    def __init__(self, qty, price_per_share, stock, strategy, action, trade_date=None):
        """
        :param trade_date: datetime, corresponding to the date and time of the trade date
        :param qty: int, number of shares traded
        :param price_per_share: float
            price per individual share in the trade or the average share price in the trade
        :param stock:
            a :class: Stock, the stock object that was traded
        :param action: str, must be *buy* or *sell* depending on what kind of trade it was
        :param position: str, must be *long* or *short*
        """
        if trade_date:
            self.trade_date = utils.parse_date(trade_date)
        else:
            self.trade_date = datetime.now()

        if action.lower() == 'buy' or action.lower() == 'sell':
            # TODO: may have to run a query to check if we own the stock or not? and if we do use update?
            self.action = action.lower()
        else:
            raise ValueError('action must be either "buy" or "sell". {} was provided.'.format(action))

        self.strategy = strategy.lower()
        self.stock = stock
        self.qty = qty
        self.price_per_share = price_per_share
        # elif position is None and corresponding_trade is not None:
        #     self.position = position
        # elif position is None and corresponding_trade is None:
        #     raise ValueError('position can only be None if a corresponding_trade is also provided and None was provided')
        # else:
        #     raise ValueError('Nice try buy, position must be either "long" or "short". {} was provided.'.format(position))
        #
        # try:
        #     self.stock = stock.make_trade(qty=qty, price_per_share=price_per_share)
        # except AttributeError:
        #     try:
        #         self.stock = OwnedStock(ticker=stock.ticker, shares_owned=qty, average_share_price=price_per_share,
        #                                 purchase_date=self.trade_date)
        #     except AttributeError:
        #         raise AttributeError('stock must be a Stock object. {} was provided'.format(type(stock)))



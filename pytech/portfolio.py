# from pytech import Session
import logging
from datetime import timedelta, datetime

import pandas_datareader.data as web
from sqlalchemy import Column, Numeric, String, DateTime, Integer
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import orm
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection

import pytech.db_utils as db
import pytech.utils as utils
from pytech.base import Base
from pytech.exceptions import AssetExistsError, InvalidActionError
from pytech.stock import Stock, OwnedAsset, Asset

logger = logging.getLogger(__name__)


class Portfolio(Base):
    """
    Holds stocks and keeps tracks of the owner's cash as well and their :class:`OwnedAssets` and allows them to perform
    analysis on just the :class:`Asset` that they currently own.


    Ignore this part:

    This class inherits from :class:``AssetUniverse`` so that it has access to its analysis functions.  It is important to
    note that :class:``Portfolio`` is its own table in the database as it represents the :class:``Asset`` that the user
    currently owns.  An :class:``Asset`` can be in both the *asset_universe* table as well as the *portfolio* table but
    a :class:``Asset`` does have to be in the database to be traded
    """

    id = Column(Integer, primary_key=True)
    cash = Column(Float)
    benchmark_ticker = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    owned_assets = relationship('OwnedAsset', backref='portfolio',
                                collection_class=attribute_mapped_collection('asset.ticker'),
                                lazy='joined', cascade='save-update, all, delete-orphan')
    assets = association_proxy('owned_assets', 'asset')

    def __init__(self, start_date=None, end_date=None, benchmark_ticker='^GSPC', starting_cash=1000000):
        """
        :param start_date: a date, the start date to the load the benchmark as of.
            (default: end_date - 365 days)
        :type start_date: datetime
        :param end_date: the end date to load the benchmark as of.
            (default: ``datetime.now()``)
        :type end_date: datetime
        :param benchmark_ticker: the ticker of the market index or benchmark to compare the portfolio against.
            (default: *^GSPC*)
        :type benchmark_ticker: str
        :param starting_cash: the amount of dollars to allocate to the portfolio initially
            (default: 10000000)
        :type starting_cash: float
        """

        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=365)
        else:
            self.start_date = utils.parse_date(start_date)

        if end_date is None:
            # default to today
            self.end_date = datetime.now()
        else:
            self.end_date = utils.parse_date(end_date)

        self.owned_assets = {}
        self.benchmark_ticker = benchmark_ticker
        self.benchmark = web.DataReader(benchmark_ticker, 'yahoo', start=self.start_date, end=self.end_date)
        self.cash = float(starting_cash)

    @orm.reconstructor
    def init_on_load(self):
        """Recreate the benchmark series on load from DB"""

        self.benchmark = web.DataReader(self.benchmark_ticker, 'yahoo', start=self.start_date, end=self.end_date)

    def add_assets_from_list(self, ticker_list, start_date, end_date=None, get_fundamentals=True, get_ohlcv=True):
        # TODO: determine if this is even needed
        for ticker in ticker_list:
            if self.owned_assets.get(ticker):
                raise AssetExistsError('Asset is already in the {} owned_assets dict.'.format(self.__class__))
            self.owned_assets[ticker] = Stock(ticker=ticker, start_date=start_date, end_date=end_date,
                                              get_fundamentals=get_fundamentals, get_ohlcv=get_ohlcv)

    def make_trade(self, ticker, qty, action, price_per_share=None, trade_date=None):
        """
        Buy or sell an asset from the asset universe.

        :param ticker: the ticker of the :class:``Asset`` to trade
        :type ticker: str
        :param qty: the number of shares to trade
        :type qty: int
        :param action: **buy** or **sell** depending on the trade
        :type action: str
        :param price_per_share: the cost per share in the trade
        :type price_per_share: long
        :param trade_date: the date and time that the trade is taking place
        :type trade_date: datetime

        This method will add the asset to the :class:``Portfolio`` asset dict and update the db to reflect the trade
        """

        try:
            asset = self.owned_assets[ticker]
            self._update_existing_position(qty=qty, action=action, price_per_share=price_per_share,
                                           trade_date=trade_date, owned_asset=asset)
        except KeyError:
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
        :raises AssetNotInUniverseError: when an asset is traded that does not yet exist in the Universe

        This method processes the trade and then writes the results to the database. It will create a new instance of
        :class:``OwnedStock`` class and at it to the :class:``Portfolio`` asset dict.
        """

        asset = Asset.get_asset_from_universe(ticker=ticker)
        if action.lower() == 'sell':
            # if selling an asset that is not in the portfolio that means it has to be a short sale.
            position = 'short'
        elif action.lower() == 'buy':
            position = 'long'
        else:
            raise InvalidActionError('action must either be "buy" or "sell". {} was provided'.format(action))

        owned_asset = OwnedAsset(asset=asset, shares_owned=qty, average_share_price=price_per_share,
                                 position=position, portfolio=self)

        self.cash += owned_asset.total_position_cost
        self.owned_assets[owned_asset.asset.ticker] = owned_asset
        trade = Trade(qty=qty, price_per_share=price_per_share, ticker=owned_asset.asset.ticker, action=action,
                      strategy='Open new {} position'.format(position), trade_date=trade_date)
        with db.transactional_session(auto_close=False) as session:
            session.add(session.merge(self))
            session.add(trade)

    def _update_existing_position(self, qty, price_per_share, trade_date, action, owned_asset):
        """
        Update the :class:``OwnedAsset`` associated with this portfolio as well as the cash position

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
        :param owned_asset: the asset that is already in the portfolio
        :type owned_asset: OwnedAsset
        :param action: **buy** or **sell**
        :type action: str

        This method processes the trade and then writes the results to the database.
        """

        owned_asset = owned_asset.make_trade(qty=qty, price_per_share=price_per_share)

        if owned_asset.shares_owned != 0:
            self.owned_assets[owned_asset.asset.ticker] = owned_asset
            self.cash += owned_asset.total_position_cost
            trade = Trade(qty=qty, price_per_share=price_per_share, ticker=owned_asset.asset.ticker,
                          strategy='Update an existing {} position'.format(owned_asset.position), action=action,
                          trade_date=trade_date)
        else:
            self.cash += owned_asset.total_position_value
            del self.owned_assets[owned_asset.asset.ticker]
            trade = Trade(qty=qty, price_per_share=price_per_share, ticker=owned_asset.asset.ticker,
                          strategy='Close an existing {} position'.format(owned_asset.position), action=action,
                          trade_date=trade_date)
        with db.transactional_session() as session:
            session.add(session.merge(self))
            session.add(trade)

    def get_total_value(self, include_cash=True):
        """
        Calculate the total value of the ``Portfolio`` owned_assets

        :param include_cash: should cash be included in the calculation, or just get the total value of the owned_assets
        :type include_cash: bool
        :return: the total value of the portfolio at a given moment in time
        :rtype: float
        """

        total_value = 0.0
        for asset in self.owned_assets.values():
            asset.update_total_position_value()
            total_value += asset.total_position_value
        if include_cash:
            total_value += self.cash
        with db.transactional_session() as session:
            # update the owned stocks in the db
            session.add(self)
        return total_value

    def portfolio_return(self):
        pass

    def sma(self):
        for ticker, stock in self.owned_assets.items():
            yield stock.simple_moving_average()


class Trade(Base):
    """
    This class is used to make trades and keep trade of past trades
    """
    id = Column(Integer, primary_key=True)
    trade_date = Column(DateTime)
    action = Column(String)
    strategy = Column(String)
    qty = Column(Integer)
    price_per_share = Column(Numeric)
    corresponding_trade_id = Column(Integer, ForeignKey('trade.id'))
    net_trade_value = Column(Numeric)
    ticker = Column(String)

    # owned_stock_id = Column(Integer, ForeignKey('owned_stock.id'))
    # owned_stock = relationship('OwnedStock')
    # corresponding_trade = relationship('Trade', remote_side=[id])

    def __init__(self, qty, price_per_share, strategy, action, trade_date=None, ticker=None):
        """
        :param trade_date: datetime, corresponding to the date and time of the trade date
        :param qty: int, number of shares traded
        :param price_per_share: float
            price per individual share in the trade or the average share price in the trade
        :param ticker:
            a :class: Stock, the asset object that was traded
        :param action: str, must be *buy* or *sell* depending on what kind of trade it was
        :param position: str, must be *long* or *short*
        """
        if trade_date:
            self.trade_date = utils.parse_date(trade_date)
        else:
            self.trade_date = datetime.now()

        if action.lower() == 'buy' or action.lower() == 'sell':
            self.action = action.lower()
        else:
            raise InvalidActionError('action must be either "buy" or "sell". {} was provided.'.format(action))

        self.strategy = strategy.lower()
        self.ticker = ticker
        self.qty = qty
        self.price_per_share = price_per_share
        self.corresponding_trade_id = self._get_corresponding_trade_id(ticker=ticker)

    @classmethod
    def _get_corresponding_trade_id(cls, ticker):
        """Get the most recent trade's id"""
        with db.query_session() as session:
            corresponding_trade = session.query(cls) \
                .filter(cls.ticker == ticker) \
                .order_by(cls.trade_date.desc()) \
                .first()
        try:
            return corresponding_trade.id
        except AttributeError:
            return None



# from pytech import Session
import logging
from datetime import datetime, timedelta

import pandas_datareader.data as web
from sqlalchemy import Column, DateTime, Float, Integer, String, orm
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection

import pytech.db_utils as db
import pytech.utils as utils
from pytech.order import Trade, Order
from pytech.base import Base
from pytech.enums import AssetPosition, TradeAction
from pytech.asset import Asset, OwnedAsset

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
    trading_cal = Column(String)
    data_frequency = Column(String)
    owned_assets = relationship('OwnedAsset', backref='portfolio',
                                collection_class=attribute_mapped_collection('asset.ticker'),
                                lazy='joined', cascade='save-update, all, delete-orphan')
    orders = relationship('Order', backref='portfolio', lazy='joined', cascade='save-update, all, delete-orphan')

    LOGGER_NAME = 'portfolio'

    # assets = association_proxy('owned_assets', 'asset')

    def __init__(self, start_date=None, end_date=None, benchmark_ticker='^GSPC', starting_cash=1000000,
                 trading_cal='NYSE', data_frequency='daily'):
        """
        :param datetime start_date: a date, the start date to start the simulation as of.
            (default: end_date - 365 days)
        :param datetime end_date: the end date to end the simulation as of.
            (default: ``datetime.now()``)
        :param str benchmark_ticker: the ticker of the market index or benchmark to compare the portfolio against.
            (default: *^GSPC*)
        :param float starting_cash: the amount of dollars to allocate to the portfolio initially
            (default: 10000000)
        :param str trading_cal: The name of the trading calendar to use.
            (default: NYSE)
        :param data_frequency: The frequency of how often data should be updated.
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

        self.trading_cal = trading_cal
        self.owned_assets = {}
        self.orders = []
        self.benchmark_ticker = benchmark_ticker
        self.benchmark = web.DataReader(benchmark_ticker, 'yahoo', start=self.start_date, end=self.end_date)
        self.cash = float(starting_cash)
        self.data_frequency = data_frequency
        self.logger = logging.getLogger(self.LOGGER_NAME)

    @orm.reconstructor
    def init_on_load(self):
        """Recreate the benchmark series on load from DB"""

        self.benchmark = web.DataReader(self.benchmark_ticker, 'yahoo', start=self.start_date, end=self.end_date)

    def make_order(self, ticker, action, order_type, stop_price=None, limit_price=None, qty=0,
                   date_placed=datetime.now(), order_subtype=None):
        """
        Open a new order.

        :param str ticker: the ticker of the :py:class:`~asset.Asset` to place an order for
        :param TradeAction or str action: **BUY** or **SELL**
        :param OrderType order_type: the type of order
        :param float stop_price: If creating a stop order this is the stop price.
        :param float limit_price: If creating a limit order this is the price that will trigger the ``order``.
        :param int qty: The number of shares to place an ``order`` for.
        :param datetime date_placed: The date and time the order is created.
        :param OrderSubType order_subtype: The type of order subtype
            (default: ``OrderSubType.DAY``)
        :param int max_days_open: Number of days to leave the ``order`` open before it expires.
        :return: None
        """

        try:
            asset = self.owned_assets[ticker]
        except KeyError:
            asset = Asset.get_asset_from_universe(ticker=ticker)

        order = Order(
            asset=asset,
            portfolio=self,
            action=action,
            order_type=order_type,
            order_subtype=order_subtype,
            stop=stop_price,
            limit=limit_price,
            qty=qty,
            created=date_placed
        )

        with db.transactional_session() as session:
            self.orders.append(order)
            session.add(self)

    def check_order_triggers(self, dt=None, current_price=None):
        """
        Check if any order has been triggered and if they have execute the trade.

        :param datetime dt: current datetime
        :param float current_price: The current price of the asset.
        """

        closed_orders = []

        for order in self.orders:
            if order.open_amount == 0 or not order.open:
                closed_orders.append(order)
                continue

            if order.check_triggers(dt=dt, current_price=current_price):
                closed_orders.append(self.make_trade(order=order, price_per_share=current_price, trade_date=dt))
                # self._process_triggered_order(order, dt=dt, current_price=current_price)

        self.purge_closed_orders(closed_orders)

    def purge_closed_orders(self, closed_orders):
        """
        Remove any order that is no longer open.

        :param iterable closed_orders:
        :return:
        """

        with db.transactional_session() as session:
            for order in closed_orders:
                if order is not None:
                    self.orders.remove(order)
                    session.delete(order)
            session.add(session.merge(self))

    def make_trade(self, order, price_per_share, trade_date):
        """
        Buy or sell an asset from the asset universe.

        :param ticker: the ticker of the :class:``Asset`` to trade
        :type ticker: str
        :param qty: the number of shares to trade
        :type qty: int
        :param action: :class:``enum.TradeAction``
        :type action: str or :class:``enum.TradeAction``
        :param float price_per_share: the cost per share in the trade
        :param trade_date: the date and time that the trade is taking place
        :type trade_date: datetime
        :return: ``order`` if the order is no longer open so it can be removed from the ``portfolio`` order dict
            and ``None`` if the order is still open
        :rtype: Order or None

        This method will add the asset to the :class:``Portfolio`` asset dict and update the db to reflect the trade.

        Valid **action** parameter values are:

        * TradeAction.BUY
        * TradeAction.SELL
        * BUY
        * SELL
        """

        try:
            asset = self.owned_assets[order.asset.ticker]
            self.logger.info('Updating {} position'.format(order.asset.ticker))
            trade = self._update_existing_position(order=order, trade_date=trade_date, owned_asset=asset,
                                           price_per_share=price_per_share)
        except KeyError:
            self.logger.info('Opening new position in {}'.format(order.asset.ticker))
            trade = self._open_new_position(order=order, price_per_share=price_per_share, trade_date=trade_date)

        order.filled += trade.qty

        if order.open:
            return None
        else:
            return order

    def _open_new_position(self, price_per_share, trade_date, order):
        """
        Create a new :py:class:``~stock.OwnedStock`` object associated with this portfolio as well as update the cash position

        :param int qty: how many shares are being bought or sold.
            If the position is a **long** position use a negative number to close it and positive to open it.
            If the position is a **short** position use a negative number to open it and positive to close it.
        :param float price_per_share: the average price per share in the trade.
            This should always be positive no matter what the trade's position is.
        :param datetime trade_date: the date and time the trade takes place
            (default: now)
        :param TradeAction or str action: either **BUY** or **SELL**
        :param Order order: The order
        :return: None
        :raises InvalidActionError: If action is not 'BUY' or 'SELL'
        :raises AssetNotInUniverseError: when an asset is traded that does not yet exist in the Universe

        This method processes the trade and then writes the results to the database. It will create a new instance of
        :py:class:`~stock.OwnedStock` class and at it to the :py:class:`~portfolio.Portfolio` asset dict.

        .. note::

        Valid **action** parameter values are:

        * TradeAction.BUY
        * TradeAction.SELL
        * BUY
        * SELL
        """

        # TODO: can I just reference the asset?
        # asset = Asset.get_asset_from_universe(ticker=order.asset.ticker)
        # action = TradeAction.check_if_valid(action)
        asset = order.asset

        if order.action is TradeAction.SELL:
            # if selling an asset that is not in the portfolio that means it has to be a short sale.
            position = AssetPosition.SHORT
            # qty *= -1
        else:
            position = AssetPosition.LONG

        owned_asset = OwnedAsset(
            asset=asset,
            shares_owned=order.get_available_volume(dt=trade_date),
            average_share_price=price_per_share,
            position=position,
            portfolio=self
        )

        self.cash += owned_asset.total_position_cost
        self.owned_assets[owned_asset.asset.ticker] = owned_asset
        trade = Trade.from_order(
            order=order,
            execution_price=price_per_share,
            trade_date=trade_date,
            strategy='Open new {} position'.format(position)
        )
        # trade = Trade(
        #     qty=qty,
        #     price_per_share=price_per_share,
        #     ticker=owned_asset.asset.ticker,
        #     action=action,
        #     strategy='Open new {} position'.format(position),
        #     trade_date=trade_date
        # )
        with db.transactional_session(auto_close=False) as session:
            session.add(session.merge(self))
            session.add(trade)
        return trade


    def _update_existing_position(self, price_per_share, trade_date, owned_asset, order):
        """
        Update the :class:``OwnedAsset`` associated with this portfolio as well as the cash position

        :param int qty: how many shares are being bought or sold.
            If the position is a **long** position use a negative number to close it and positive to open it.
            If the position is a **short** position use a negative number to open it and positive to close it.
        :param float price_per_share: the average price per share in the trade.
            This should always be positive no matter what the trade's position is.
        :param datetime trade_date: the date and time the trade takes place
            (default: now)
        :param OwnedAsset owned_asset: the asset that is already in the portfolio
        :param TradeAction action: **BUY** or **SELL**
        :param Order order:
        :raises InvalidActionError:

        This method processes the trade and then writes the results to the database.
        """
        # action = TradeAction.check_if_valid(action)

        # if order.action is TradeAction.SELL:
        #     qty *= -1

        owned_asset = owned_asset.make_trade(qty=order.get_available_volume(dt=trade_date),
                                             price_per_share=price_per_share)

        if owned_asset.shares_owned != 0:
            self.owned_assets[owned_asset.asset.ticker] = owned_asset
            self.cash += owned_asset.total_position_cost

            trade = Trade.from_order(
                order=order,
                execution_price=price_per_share,
                trade_date=trade_date,
                strategy='Update an existing {} position'.format(owned_asset.position)
            )

            # trade = Trade(
            #     qty=qty,
            #     price_per_share=price_per_share,
            #     ticker=owned_asset.asset.ticker,
            #     strategy='Update an existing {} position'.format(owned_asset.position),
            #     action=action,
            #     trade_date=trade_date
            # )
        else:
            self.cash += owned_asset.total_position_value

            del self.owned_assets[owned_asset.asset.ticker]

            trade = Trade.from_order(
                order=order,
                execution_price=price_per_share,
                trade_date=trade_date,
                strategy='Close an existing {} position'.format(owned_asset.position)
            )

            # trade = Trade(
            #     qty=qty,
            #     price_per_share=price_per_share,
            #     ticker=owned_asset.asset.ticker,
            #     strategy='Close an existing {} position'.format(owned_asset.position),
            #     action=action,
            #     trade_date=trade_date
            # )

        with db.transactional_session() as session:
            session.add(session.merge(self))
            session.add(trade)

        return trade

    def get_total_value(self, include_cash=True):
        """
        Calculate the total value of the ``Portfolio`` owned_assets

        :param include_cash:
            should cash be included in the calculation, or just get the total value of the owned_assets
        :type include_cash: bool
        :return:
            the total value of the portfolio at a given moment in time
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

    def return_on_owned_assets(self):
        """Get the total return of the portfolio's current owned assets"""

        roi = 0.0
        for asset in self.owned_assets.values():
            roi += asset.return_on_investment()
        return roi

    def sma(self):
        for ticker, stock in self.owned_assets.items():
            yield stock.simple_moving_average()

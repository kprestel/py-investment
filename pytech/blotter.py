from datetime import datetime

import logging
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection

from pytech import Base, utils
from pytech.asset import Asset, OwnedAsset
from pytech.enums import AssetPosition
from pytech.enums import TradeAction
from pytech.exceptions import NotAPortfolioError
from pytech.order import Order, Trade
import pytech.db_utils as db


class Blotter(Base):
    """Holds and interacts with all orders."""

    id = Column(Integer, primary_key=True)
    portfolio = relationship('Portfolio', back_populates='blotter')
    portfolio_id = Column(Integer, ForeignKey('portfolio.id'))
    orders = relationship('Order', backref='blotter',
                          collection_class=attribute_mapped_collection('asset.ticker'),
                          lazy='joined', cascade='save-update, all, delete-orphan')

    def __init__(self, portfolio):

        # feels hacky but idk how else to avoid circular dependency
        from pytech.portfolio import Portfolio

        if not isinstance(portfolio, Portfolio):
            raise NotAPortfolioError(type(portfolio))

        self.portfolio = portfolio
        # dict of all orders. key is the ticker of the asset, value is the asset
        self.orders = {}
        self.current_dt = None
        # TODO: reference portfolio in the logger name
        self.logger = logging.getLogger('blotter')


    def place_order(self, ticker, action, order_type, stop_price=None, limit_price=None, qty=0,
                   date_placed=None, order_subtype=None):
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

        asset = self.portfolio.get_owned_asset(ticker)

        if asset is None:
            asset = Asset.get_asset_from_universe(ticker=ticker)

        if date_placed is None:
            date_placed = self.current_dt

        order = Order(
                asset=asset,
                blotter=self,
                action=action,
                order_type=order_type,
                order_subtype=order_subtype,
                stop=stop_price,
                limit=limit_price,
                qty=qty,
                created=date_placed
        )

        with db.transactional_session() as session:
            self.orders[ticker] = order
            session.add(self)

    def check_order_triggers(self, dt=None, current_price=None):
        """
        Check if any order has been triggered and if they have execute the trade.

        :param datetime dt: current datetime
        :param float current_price: The current price of the asset.
        """

        closed_orders = []

        for order in self.orders.values():
            if order.open_amount == 0 or not order.open:
                closed_orders.append(order)
                continue

            if order.check_triggers(dt=dt, current_price=current_price):
                # make_trade will return the order if it closed
                closed_orders.append(self.make_trade(order=order, price_per_share=current_price, trade_date=dt))

        self.purge_closed_orders(closed_orders)

    def make_trade(self, order, price_per_share, trade_date):
        """
        Buy or sell an asset from the asset universe.

        :param str ticker: The ticker of the :class:``Asset`` to trade.
        :param int qty: the number of shares to trade
        :param TradeAction or str action: :py:class:``enum.TradeAction`` see comments below.
        :param float price_per_share: the cost per share in the trade
        :param datetime trade_date: The date and time that the trade is taking place.
        :return: ``order`` if the order is no longer open so it can be removed from the ``portfolio`` order dict
            and ``None`` if the order is still open
        :rtype: Order or None

        This method will add the asset to the :py:class:``Portfolio`` asset dict and update the db to reflect the trade.

        Valid **action** parameter values are:

        * TradeAction.BUY
        * TradeAction.SELL
        * BUY
        * SELL
        """

        if order.owned_asset:
            self.logger.info('Updating {} position'.format(order.asset.ticker))
            trade = self._update_existing_position(order=order, trade_date=trade_date, owned_asset=order.owned_asset,
                                                   price_per_share=price_per_share)
        else:
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
                portfolio=self.portfolio
        )

        self.portfolio.cash += owned_asset.total_position_cost
        self.portfolio.owned_assets[owned_asset.asset.ticker] = owned_asset

        trade = Trade.from_order(
                order=order,
                execution_price=price_per_share,
                trade_date=trade_date,
                strategy='Open new {} position'.format(position)
        )

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
        else:
            self.portfolio.cash += owned_asset.total_position_value

            del self.portfolio.owned_assets[owned_asset.asset.ticker]

            trade = Trade.from_order(
                    order=order,
                    execution_price=price_per_share,
                    trade_date=trade_date,
                    strategy='Close an existing {} position'.format(owned_asset.position)
            )

        with db.transactional_session() as session:
            session.add(session.merge(self))
            session.add(trade)

        return trade

    def purge_closed_orders(self, closed_orders):
        """
        Remove any order that is no longer open.

        :param iterable closed_orders:
        :return:
        """

        with db.transactional_session() as session:
            for order in closed_orders:
                if order is not None:
                    del self.orders[order.asset.ticker]
                    session.delete(order)
            session.add(session.merge(self))



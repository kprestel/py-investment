import logging
from datetime import datetime

import collections

import pytech.db.db_utils as db
from pytech.db.finders import AssetFinder
from pytech.fin.owned_asset import OwnedAsset
from pytech.fin.asset import Asset
from pytech.fin.portfolio import Portfolio
from pytech.trading.order import Order, Trade
from pytech.utils.enums import AssetPosition, TradeAction
from pytech.utils.exceptions import NotAFinderError, NotAPortfolioError
from pytech.trading.commission import PerOrderCommissionModel


class Blotter(object):
    """Holds and interacts with all orders."""

    LOGGER_NAME = 'blotter'

    def __init__(self, asset_finder=None, portfolio=None, commission_model=None):

        self.logger = logging.getLogger(self.LOGGER_NAME)
        self.asset_finder = asset_finder
        self.portfolio = portfolio
        # dict of all orders. key is the ticker of the asset, value is the asset.
        self.orders = {}
        # keep a record of all past trades.
        self.trades = []
        self.current_dt = None
        self.commission_model = commission_model or PerOrderCommissionModel

    @property
    def asset_finder(self):
        return self._asset_finder

    @asset_finder.setter
    def asset_finder(self, asset_finder):
        if asset_finder is not None and isinstance(asset_finder, AssetFinder):
            self._asset_finder = asset_finder
        elif asset_finder is None:
            self._asset_finder = AssetFinder()
        else:
            raise NotAFinderError(finder=type(asset_finder))

    @property
    def portfolio(self):
        return self._portfolio

    @portfolio.setter
    def portfolio(self, portfolio):
        if portfolio is not None and isinstance(portfolio, Portfolio):
            self._portfolio = portfolio
        elif portfolio is None:
            self._portfolio = Portfolio()
        else:
            raise NotAPortfolioError(portfolio=type(portfolio))

    def __getitem__(self, key):
        """Get an order from the orders dict."""

        return self.orders[key]

    def __setitem__(self, key, value):
        """
        Add an order to the orders dict.
        If the key is an instance of :class:`~asset.Asset` then the ticker is used as the key, otherwise the key is the
        ticker.
        :param key: The key to dictionary, will always be the ticker of the ``Asset`` the order is for but an instance
        of :class:`~asset.Asset` will also work as long as the ticker is set.
        :type key: Asset or str
        :param Order value: The order.
        """

        if issubclass(key.__class__, Asset):
            self.orders[key.ticker] = value
        else:
            self.orders[key] = value

    def __delitem__(self, key):
        """Delete an order from the orders dict."""

        del self.orders[key]

    def __iter__(self):
        """
        Iterate over the orders dict as well as the nested orders dict which key=order_id and value=``Order``
        This means you can iterate over a :class:``Blotter`` instance directly and access all of the open orders it has.
        """

        def do_iter(orders_dict):
            for k, v in orders_dict.items():
                if isinstance(v, collections.Mapping):
                    yield from do_iter(v)
                else:
                    yield k, v

        return do_iter(self.orders)

    def place_order(self, ticker, action, order_type, qty, stop_price=None, limit_price=None,
                    date_placed=None, order_subtype=None, order_id=None):
        """
        Open a new order.  If an open order for the given ``ticker`` already exists placing a new order will **NOT**
        change the existing order, it will be added to the tuple.

        :param ticker: The asset of the :py:class:`~asset.Asset` to place an order for or the ticker of an asset.
        :type ticker: Asset or str
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

        if qty == 0:
            # No point in making an order for 0 shares.
            return None

        if date_placed is None:
            date_placed = self.current_dt

        order = Order(
                asset=ticker,
                action=action,
                order_type=order_type,
                order_subtype=order_subtype,
                stop=stop_price,
                limit=limit_price,
                qty=qty,
                created=date_placed,
                id=order_id
        )

        if ticker in self.orders:
            self.orders[ticker].update({order.id: order})
        else:
            self.orders[ticker] = {order.id: order}

    def _find_order(self, order_id, ticker):

        if ticker is None:
            for asset_orders in self.orders:
                if order_id in asset_orders:
                    return asset_orders[order_id]
        else:
            for k, v in self.orders[ticker].items():
                if k == order_id:
                    return v

    def cancel_order(self, order_id, ticker=None, reason=''):
        """
        Mark an order as canceled so that it will not get executed.

        :param str order_id: The id of the order to cancel.
        :param ticker: (optional) The ticker that the order is associated with.
            Although it is not required to provide a ticker, it is **strongly** encouraged.
            By providing a ticker the execution time of this method will increase greatly.
        :param str reason: (optional) The reason that the order is being cancelled.
        :return:
        """

        self._do_order_cancel(self._find_order(order_id, ticker), reason)

    def cancel_all_orders_for_asset(self, ticker, reason=''):
        """
        Cancel all orders for a given asset's ticker and then clean up the orders dict.

        :param str ticker: The ticker of the asset to cancel all orders for.
        :param str reason: (optional) The reason for canceling the order.
        :return:
        """

        for order in self.orders[ticker].values():
            self._do_order_cancel(order, reason)

    def _do_order_cancel(self, order, reason):
        """Cancel any order that is passed to this method and log the appropriate message."""

        if order.filled > 0:
            self.logger.warning('Order for {ticker} has been partially filled.'
                                '{amt} shares have already been successfully purchased.'
                                .format(ticker=order.asset, amt=order.filled))
        elif order.filled < 0:
            self.logger.warning('Order for {ticker} has been partially filled.'
                                '{amt} shares have already been successfully sold.'
                                .format(ticker=order.asset, amt=order.filled))
        else:
            self.logger.info('Canceled order for {ticker} successfully before it was executed.'
                             .format(ticker=order.asset))
        order.cancel(reason)
        order.last_updated = self.current_dt

    def reject_order(self, order_id, ticker=None, reason=''):
        """
        Mark an order as rejected. A rejected order is functionally the same as a canceled order but an order being
        marked rejected is typically involuntary or unexpected and comes from the broker. Another case that an order
        will be rejected is if when the order is being executed the owner does not have enough cash to fully execute it.

        :param str order_id: The id of the order being rejected.
        :param str ticker: (optional) The ticker associated with the order being rejected.
        :param str reason: (optional) The reason the order was rejected.
        :return:
        """

        self._find_order(order_id, ticker).reject(reason)

        self.logger.warning('Order id: {id} for ticker: {ticker} was rejected because: {reason}'
                            .format(id=order_id, ticker=ticker, reason=reason or 'Unknown'))

    def check_order_triggers(self, dt=None, current_price=None):
        """
        Check if any order has been triggered and if they have execute the trade and then clean up closed orders.

        :param datetime dt: current datetime
        :param float current_price: The current price of the asset.
        """

        for order in self.orders.values():
            if order.check_triggers(dt=dt, current_price=current_price):
                # make_trade will return the order if it closed
                self.make_trade(order, current_price, dt)

        self.purge_orders()

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

        commission_cost = self.commission_model.calculate(order, price_per_share)

        order.commission += commission_cost

        if self.portfolio.check_liquidity(commission_cost, price_per_share, order.qty):
            trade = Trade.from_order(order, trade_date, price_per_share, commission_cost, 'Executing Order.')
            order.filled += trade.qty
            self.portfolio.cash += trade.trade_value()
            self.trades.append(trade)
        else:
            order.reject('Insufficient Funds to make trade.  Order was rejected.')
            self.logger.warning('Not enough cash to place the trade.')

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
                ticker=asset,
                shares_owned=order.get_available_volume(dt=trade_date),
                average_share_price=price_per_share,
                position=position
        )

        self.portfolio.cash += owned_asset.total_position_cost
        self.portfolio[asset] = owned_asset

        trade = Trade.from_order(
                order=order,
                execution_price=price_per_share,
                trade_date=trade_date,
                strategy='Open new {} position'.format(position)
        )

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
        """

        owned_asset = owned_asset.make_trade(qty=order.get_available_volume(dt=trade_date),
                                             price_per_share=price_per_share)

        if owned_asset.shares_owned != 0:
            self.portfolio[owned_asset.asset.ticker] = owned_asset
            self.portfolio.cash += owned_asset.total_position_cost

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

        return trade

    def purge_orders(self):
        """Remove any order that is no longer open."""

        open_orders = {}

        for ticker, asset_orders in self.orders:
            for order in asset_orders:
                if order.open and order.open_amount != 0:
                    open_orders[ticker] = order

        self.orders = open_orders

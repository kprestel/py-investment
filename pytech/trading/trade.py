"""
Trade module which contains anything related to the actual execution of a trade.
"""
import logging
from datetime import datetime

import pandas as pd

from pytech.utils import dt_utils as dt_utils
from pytech.utils.enums import TradeAction
from pytech.utils.exceptions import UntriggeredTradeError


class Trade(object):
    """
    This class is used to make trades and keep trade of past trades.

    Trades must be created as a result of an :class:``Order`` executing.
    """

    LOGGER_NAME = 'trade'

    def __init__(self, qty, price_per_share, action, strategy, order,
                 avg_price_per_share, commission=0.0,
                 trade_date=None, ticker=None):
        """
        :param datetime trade_date: corresponding to the date and time of the 
            trade date
        :param int qty: number of shares traded
        :param float price_per_share: price per individual share in the trade 
            or the average share price in the trade
        :param Asset ticker: a :py:class:`~.ticker.Asset`, the ``ticker`` 
            object that was traded
        :param Order order: a :py:class:`~.order.Order` that was executed as a 
            result of the order executing
        :param float commission: the amount of commission paid to execute this trade
            (default: 0.0)
        :param TradeAction or str action: :py:class:`~.enum.TradeAction`
        :param str position: must be *long* or *short*

        .. note::

        Valid **action** parameter values are:

        * TradeAction.BUY
        * TradeAction.SELL
        * BUY
        * SELL

        `commission` is not necessarily the same commission associated with the 
        ``order`` this will depend on the
        type of :class:``AbstractCommissionModel`` used.
        """

        if trade_date:
            self.trade_date = dt_utils.parse_date(trade_date)
        else:
            self.trade_date = pd.Timestamp(datetime.now())

        self.action = TradeAction.check_if_valid(action)
        self.strategy = strategy
        self.ticker = ticker
        self.qty = qty
        self.price_per_share = price_per_share
        self.order = order
        self.commission = commission
        self.avg_price_per_share = avg_price_per_share
        self.logger = logging.getLogger(self.LOGGER_NAME)

    def trade_cost(self):
        """
        Return the total financial impact of a trade. 
        
        If this was a buy order then the impact will be negative.
        """

        return (self.qty * self.price_per_share) + self.commission

    def trade_value(self):
        return self.trade_value() * -1

    @classmethod
    def from_order(cls, order, trade_date, commission, price_per_share, qty,
                   avg_price_per_share, strategy=None):
        """
        Make a trade from a triggered order object.

        :param Order order: The ``order`` object creating the trade.
        :param datetime trade_date: The datetime that the trade was executed.
        :param float commission: The amount of commission charged on the trade.
        :param float price_per_share: The true price per share, i.e. 
            before commission has been added.
        :param int qty: The number of shares that the trade is for.
        :param float avg_price_per_share: The average price per share in the 
            trade AFTER commission has been applied.
        :param str strategy: (optional) The strategy of the trade.
        :raises UntriggeredTradeError: if the order has not been triggered.
        :return: a new ``trade``
        :rtype: Trade
        """

        if not order.triggered:
            raise UntriggeredTradeError(order=order.__dict__.__str__())

        if strategy is None:
            strategy = order.order_type.name

        trade_dict = {
            'ticker': order.ticker,
            'qty': qty,
            'action': order.action,
            'price_per_share': price_per_share,
            'avg_price_per_share': avg_price_per_share,
            'strategy': strategy,
            'order': order,
            'trade_date': trade_date,
            'commission': commission
        }

        return cls(**trade_dict)

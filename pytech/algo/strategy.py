import logging
from abc import ABCMeta, abstractmethod

import pytech.utils.pandas_utils as pd_utils
from pytech.backtest.event import (MarketEvent, SignalEvent)
from pytech.data.handler import DataHandler
from pytech.trading.order import get_order_types
from pytech.utils.enums import EventType

OrderTypes = get_order_types()


class Strategy(metaclass=ABCMeta):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def generate_signals(self, event):
        """Provides the mechanisms to calculate a list of signals."""

        raise NotImplementedError('Must implement generate_signals()')


class BuyAndHold(Strategy):
    def __init__(self, data_handler, events):
        super().__init__()

        if not issubclass(data_handler.__class__, DataHandler):
            raise TypeError(
                    'bars must be a subclass of DataHandler. '
                    '{} was provided'.format(type(data_handler))
            )
        else:
            self.bars = data_handler

        self.ticker_list = self.bars.tickers
        self.events = events
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dict for all symbols and sets them to false.
        """
        bought = {}

        for ticker in self.ticker_list:
            bought[ticker] = False

        return bought

    def generate_signals(self, event: MarketEvent):
        """
        For buy and hold we generate a single single per symbol and then 
        no more signals. Meaning we are always long the market 
        from the start to the end of the sim.

        :param MarketEvent event: A :class:`MarketEvent` object.
        """
        self.logger.info('Generating signals')

        if event.event_type is EventType.MARKET:

            for ticker in self.ticker_list:
                self.logger.debug(
                        f'Processing ticker: {ticker}')
                bars = self.bars.get_latest_bar_value(
                        ticker, pd_utils.ADJ_CLOSE_COL)
                self.logger.debug(f'bars: {bars}')

                if bars is not None:
                    if not self.bought[ticker]:
                        signal = SignalEvent(ticker, 'LONG', bars[0])
                        self.events.put(signal)
                        self.bought[ticker] = True

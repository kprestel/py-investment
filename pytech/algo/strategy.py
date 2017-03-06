from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from queue import Queue
from pytech.backtest.event import SignalEvent, MarketEvent
from pytech.utils.enums import EventType
from pytech.data.handler import YahooDataHandler, DataHandler


class Strategy(metaclass=ABCMeta):

    @abstractmethod
    def generate_signals(self, event):
        """Provides the mechanisms to calculate a list of signals."""

        raise NotImplementedError('Must implement generate_signals()')


class BuyAndHold(Strategy):

    def __init__(self, data_handler, events):

        if not issubclass(data_handler.__class__, DataHandler):
            raise TypeError('bars must be a subclass of DataHandler. {} was provided'
                            .format(type(data_handler)))
        else:
            self.data_handler = data_handler

        self.ticker_list = self.data_handler.ticker_list
        self.events = events
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """Adds keys to the bought dict for all symbols and sets them to false."""

        bought = {}

        for ticker in self.ticker_list:
            bought[ticker] = False

        return bought

    def generate_signals(self, event):
        """
        For buy and hold we generate a single single per symbol and then no more signals.  Meaning we are always long
        the market from the start to the end of the sim.

        :param MarketEvent event: A :class:`MarketEvent` object.
        """

        if event.type is EventType.MARKET:

            for ticker in self.ticker_list:
                bars = self.data_handler.get_latest_bars(ticker)

                if bars is not None:
                    if not self.bought[ticker]:
                        signal = SignalEvent(ticker, bars[1], 'LONG')
                        self.events.put(signal)
                        self.bought[ticker] = True





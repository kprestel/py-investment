import logging
from abc import ABCMeta, abstractmethod
from queue import Queue

import pandas as pd
from queuelib import queue

import pytech.utils.pandas_utils as pd_utils
from pytech.backtest.event import MarketEvent, SignalEvent
from pytech.data.handler import DataHandler
from pytech.trading.order import get_order_types
from pytech.utils.enums import EventType, Position, SignalType, TradeAction
from pytech.utils.exceptions import InvalidEventTypeError

OrderTypes = get_order_types()


class Strategy(metaclass=ABCMeta):
    def __init__(self, data_handler: DataHandler, events):
        self.logger = logging.getLogger(__name__)

        if not issubclass(data_handler.__class__, DataHandler):
            raise TypeError('bars must be a subclass of DataHandler. '
                            f'{type(data_handler)} was provided')
        else:
            self.bars = data_handler

        self.ticker_list = self.bars.tickers
        self.events = events

    @abstractmethod
    def generate_signals(self, event):
        """Provides the mechanisms to calculate a list of signals."""

        raise NotImplementedError('Must implement generate_signals()')


class BuyAndHold(Strategy):
    def __init__(self, data_handler, events):
        super().__init__(data_handler, events)

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
                self.logger.debug(f'Processing ticker: {ticker}')
                bars = self.bars.get_latest_bar_value(ticker,
                                                      pd_utils.ADJ_CLOSE_COL)

                if bars is not None:
                    if not self.bought[ticker]:
                        signal = SignalEvent(ticker, 'LONG', bars[0])
                        self.events.put(signal)
                        self.bought[ticker] = True


class CrossOverStrategy(Strategy):
    def __init__(self, data_handler: DataHandler,
                 events: Queue,
                 short_window: int = 50,
                 long_window: int = 200):
        super().__init__(data_handler, events)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, event: MarketEvent):
        """

        :param event:
        :return:
        """
        self.logger.info('Generating signals')

        if event.event_type is not EventType.MARKET:
            raise InvalidEventTypeError(expected=EventType.MARKET,
                                        event_type=event.event_type)

        for ticker in self.ticker_list:
            bar = self.bars.get_latest_bar_value(ticker,
                                                 pd_utils.ADJ_CLOSE_COL,
                                                 n=self.long_window)
            signals = pd.DataFrame(bar)

            signals['short_mavg'] = pd.rolling_mean(
                    bar,
                    self.short_window,
                    min_periods=1)
            signals['long_mavg'] = pd.rolling_mean(bar,
                                                   self.long_window,
                                                   min_periods=1)
            short = signals['short_mavg'].tail(1)
            long = signals['long_mavg'].tail(1)
            short = short.iat[0]
            long = long.iat[0]
            self.logger.debug(
                    f'Ticker: {ticker}, long: {long}, short: {short}')

            # TODO: make me smarter.
            if short > long:
                self.logger.debug(f'Creating LONG signal for ticker: {ticker}')
                self.events.put(
                        SignalEvent(
                                ticker,
                                signal_type=SignalType.LONG,
                                stop_price=short,
                                action=TradeAction.BUY,
                                position=Position.LONG))
            elif short < long:
                self.logger.debug(f'Creating SHORT signal for ticker: {ticker}')
                self.events.put(
                        SignalEvent(
                                ticker,
                                signal_type=SignalType.SHORT,
                                stop_price=long,
                                action=TradeAction.SELL,
                                position=Position.SHORT))
            else:
                continue

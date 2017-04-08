import datetime as dt
import logging
import queue
import pytech.utils.dt_utils as dt_utils
import pytech.utils.common_utils as com_utils

from pytech.data.handler import YahooDataHandler
from pytech.trading.blotter import Blotter
from pytech.trading.execution import SimpleExecutionHandler
from pytech.fin.portfolio import BasicPortfolio
from pytech.utils.enums import EventType


class Backtest(object):
    """
    Does backtest stuff...

    update me.
    """

    def __init__(self, ticker_list, initial_capital, start_date, strategy,
                 end_date=None, data_handler=None, execution_handler=None,
                 portfolio=None):
        """
        Initialize the backtest.

        :param iterable ticker_list: A list of tickers.
        :param initial_capital: Amount of starting capital.
        :param start_date: The date to start the backtest as of.
        :param strategy: The strategy to backtest.
        :param data_handler:
        :param execution_handler:
        :param portfolio:
        """
        self.logger = logging.getLogger(__name__)
        self.ticker_list = com_utils.iterable_to_set(ticker_list)
        self.initial_capital = initial_capital
        self.start_date = dt_utils.parse_date(start_date)

        if end_date is None:
            self.end_date = dt.datetime.utcnow()
        else:
            self.end_date = dt_utils.parse_date(end_date)
        self.strategy_cls = strategy

        if data_handler is None:
            self.data_handler_cls = YahooDataHandler
        else:
            self.data_handler_cls = data_handler

        if execution_handler is None:
            self.execution_handler_cls = SimpleExecutionHandler
        else:
            self.execution_handler_cls = execution_handler

        if portfolio is None:
            self.portfolio_cls = BasicPortfolio
        else:
            self.portfolio_cls = portfolio

        self.events = queue.Queue()

        self.blotter = Blotter(self.events)

        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

        self._init_trading_instances()

    def _init_trading_instances(self):
        self.data_handler = self.data_handler_cls(self.events,
                                                  self.ticker_list,
                                                  self.start_date,
                                                  self.end_date)
        self.blotter.bars = self.data_handler
        self.strategy = self.strategy_cls(self.data_handler, self.events)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events,
                                            self.start_date, self.blotter,
                                            self.initial_capital)
        self.execution_handler = self.execution_handler_cls(self.events)

    def _run(self):
        iterations = 0

        while True:
            iterations += 1
            self.logger.info('Iteration #{}'.format(iterations))

            if self.data_handler.continue_backtest:
                self.logger.debug('Updating bars.')
                self.data_handler.update_bars()
            else:
                self.logger.info('Backtest completed.')
                break

            # handle events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    self.logger.info('Event queue is empty. '
                                     'Continuing to next day.')
                    break
                else:
                    if event is not None:
                        self._process_event(event)

    def _process_event(self, event):
        self.logger.info(
                'Processing {event_type}'.format(event_type=event.type))

        if event.type is EventType.MARKET:
            self.strategy.generate_signals(event)
            self.portfolio.update_timeindex(event)
        elif event.type is EventType.SIGNAL:
            self.signals += 1
            self.portfolio.update_signal(event)
        elif event.type is EventType.TRADE:
            self.orders += 1
            self.execution_handler.execute_order(event)
        elif event.type is EventType.FILL:
            self.fills += 1
            self.portfolio.update_fill(event)
        else:
            return

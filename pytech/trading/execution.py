import datetime as dt
from abc import ABCMeta, abstractmethod
from pytech.utils.enums import EventType
from pytech.backtest.event import FillEvent, TradeAction


class ExecutionHandler(metaclass=ABCMeta):
    """
    UPDATE ME
    """

    @abstractmethod
    def execute_order(self, event):
        """Execute the order"""

        raise NotImplementedError('Implement execute_order you dummy.')


class SimpleExecutionHandler(ExecutionHandler):

    def __init__(self, events):

        self.events = events

    def execute_order(self, event):

        if event.type is EventType.TRADE:
            fill_event = FillEvent(dt.datetime.utcnow(), event.ticker, 'NYSE', event.qty, event.action, None)

            self.events.put(fill_event)

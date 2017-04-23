import pytest
from pytech.backtest.event import (Event, MarketEvent, SignalEvent, TradeEvent,
                                   FillEvent)

class TestMarketEvent(object):

    market_event = MarketEvent()

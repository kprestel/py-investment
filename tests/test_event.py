import pytest
from pytech.backtest.event import (Event, MarketEvent, SignalEvent, TradeEvent,
                                   FillEvent)
from pytech.utils.enums import EventType, SignalType


class TestMarketEvent(object):

    def test_market_event(self):
        market_event = MarketEvent()
        assert market_event.event_type is EventType.MARKET
        assert issubclass(market_event.__class__, Event)

    def test_signal_event(self):
        signal_event = SignalEvent('AAPL', '2017-03-18', SignalType.LONG,
                                   target_price=101.22)
        assert signal_event.event_type is EventType.SIGNAL
        assert signal_event.signal_type is SignalType.LONG
        assert signal_event.ticker == 'AAPL'
        assert signal_event.target_price == 101.22
        assert signal_event.order_type is None
        assert isinstance(signal_event, SignalEvent)
        assert issubclass(signal_event.__class__, Event)

    def test_trade_event(self):
        trade_event = TradeEvent('one', 111.11, 2, '2017-03-18')
        assert trade_event.event_type is EventType.TRADE
        assert trade_event.qty == 2
        assert issubclass(trade_event.__class__, Event)

    def test_fill_event(self):
        fill_event = FillEvent('one', 112.11, 500, '2017-03-18')
        assert fill_event.event_type is EventType.FILL
        assert fill_event.price == 112.11
        assert fill_event.available_volume == 500
        assert issubclass(fill_event.__class__, Event)

    def test_event_from_dict(self):

        signal_event_dict = {
            'ticker': 'AAPL',
            'dt': '2017-03-18',
            'signal_type': 'SHORT',
            'limit_price': 124.11,
            'upper_price': 160.11,
            'action': 'SELL',
            'junk': 'more junk'
        }

        signal_event = SignalEvent.from_dict(signal_event_dict)
        assert isinstance(signal_event, SignalEvent)
        assert signal_event.ticker == 'AAPL'
        assert signal_event.signal_type is SignalType.SHORT


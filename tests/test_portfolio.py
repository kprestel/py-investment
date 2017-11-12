from typing import Dict

import numpy as np
import pytest
from fin.asset.owned_asset import OwnedAsset
from pytech.backtest.event import MarketEvent
from pytech.fin.portfolio import BasicPortfolio
from pytech.fin.portfolio.portfolio import (
    Portfolio,
)
from utils.enums import Position


class TestBasicPortfolio(object):
    def test_constructor(self, bars, events, blotter,
                         date_range):
        test_portfolio = BasicPortfolio(bars, events,
                                        date_range, blotter)
        assert test_portfolio is not None

    def test_update_timeindex(self, basic_portfolio):
        """
        Test updating the time index.

        :param BasicPortfolio basic_portfolio:
        """
        basic_portfolio.update_timeindex(MarketEvent())
        assert basic_portfolio.owned_assets == {}
        assert basic_portfolio.all_holdings_mv[0]['AAPL'] == 0.0
        basic_portfolio.update_timeindex(MarketEvent())

    # noinspection PyTypeChecker
    def test_current_weights(self, basic_portfolio: Portfolio):
        """Test the calculating of portfolio weights"""

        # noinspection PyTypeChecker
        def total_is_one(weights: Dict[str, float]):
            total = 0
            for v in weights.values():
                total += v
            assert np.isclose(total, 1.0)

        owned_assets = {
            'AAPL': OwnedAsset('AAPL',
                               shares_owned=100,
                               position=Position.LONG,
                               avg_share_price=100.00),
            'FB': OwnedAsset('FB', shares_owned=100,
                             position=Position.LONG,
                             avg_share_price=100.00)
        }
        basic_portfolio.owned_assets = owned_assets
        basic_portfolio.cash = 10_000

        weights = basic_portfolio.current_weights()
        assert weights.get('AAPL') == .5
        assert weights.get('FB') == .5
        total_is_one(weights)

        weights = basic_portfolio.current_weights(include_cash=True)
        assert np.isclose(weights.get('AAPL'), .333333)
        assert np.isclose(weights.get('FB'), .333333)
        assert np.isclose(weights.get('cash'), .333333)
        total_is_one(weights)

        msft = OwnedAsset('MSFT',
                          shares_owned=100,
                          position=Position.SHORT,
                          avg_share_price=100.00)
        basic_portfolio.owned_assets.update({'MSFT': msft})
        weights = basic_portfolio.current_weights()
        assert np.isclose(weights.get('AAPL'), 1)
        assert np.isclose(weights.get('FB'), 1)
        assert np.isclose(weights.get('MSFT'), -1)
        total_is_one(weights)

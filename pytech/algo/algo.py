from abc import ABCMeta

from pytech.db.enviornment import Environment
from pytech.db.finders import AssetFinder
from pytech.fin.portfolio import Portfolio
from pytech.trading.blotter import Blotter
import pytech.utils.dt_utils as dt_utils


class Algorithim(metaclass=ABCMeta):
    """Abstract Base Class that all trading Algorithms must inherit from."""

    def __init__(self, *args, **kwargs):

        self.start_date = kwargs.pop('start_date', dt_utils.get_default_date(is_start_date=True))
        self.end_date = kwargs.pop('end_date', dt_utils.get_default_date(is_start_date=False))
        self.environment = kwargs.pop('env', Environment())
        self.asset_finder = kwargs.pop('asset_finder', self.environment.asset_finder)
        self.portfolio = kwargs.pop('portfolio', Portfolio())
        self.blotter = kwargs.pop('blotter')

        if self.blotter is None:
            self.blotter = Blotter(
                asset_finder=self.asset_finder,
                portfolio=self.portfolio
            )

        self.sim_params = kwargs.pop('sim_params')


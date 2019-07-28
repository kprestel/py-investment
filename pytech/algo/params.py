import pytech.utils.dt_utils as dt_utils
from pytech.fin.portfolio import Portfolio
from pytech.trading.commission import PerOrderCommissionModel, AbstractCommissionModel
from pytech.trading.blotter import Blotter


class AlgoParams(object):
    """
    Sets all parameters place restrictions on how the actual :class:`Algorithm` will run.

    These include:
        * Start Date
        * End Date
        * Trade Restrictions
        * Cancel Policies
        * Broker Policies
        * Commission Policies
        * Blacklisted Stocks
        * Initial Capital
    """

    def __init__(
        self, starting_capital, start_date=None, end_date=None, commission_model=None
    ):

        self.portfolio = Portfolio(starting_capital)
        self.start_date = start_date or dt_utils.get_default_date(True)
        self.end_date = end_date or dt_utils.get_default_date(False)
        self.blotter = Blotter(
            portfolio=self.portfolio, commission_model=commission_model
        )

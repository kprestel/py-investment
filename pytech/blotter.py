from pytech import Base, utils
from pytech.exceptions import NotAPortfolioError
from pytech.portfolio import Portfolio

class Blotter(Base):
    """Holds and interacts with all orders."""

    def __init__(self, portfolio):

        if not isinstance(portfolio, Portfolio):
            raise NotAPortfolioError(type(portfolio))

        self.portfolio = portfolio

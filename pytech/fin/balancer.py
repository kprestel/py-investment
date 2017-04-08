import logging
import cvxpy as cvx
from abc import ABCMeta, abstractmethod
from pytech.fin.portfolio import AbstractPortfolio


class AbstractBalancer(metaclass=ABCMeta):
    """Base class for all balancers."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def balance(self, portfolio):
        """
        Balance the portfolio based on whatever methodology chosen.
        
        :param Portfolio portfolio: 
        :return: 
        """
        raise NotImplementedError('Must implement balance(portfolio)')


class ClassicalBalancer(AbstractBalancer):
    """Balance based on Markowitz portfolio optimization."""

    def __init__(self):
        super().__init__()

    def balance(self, portfolio):
        pass


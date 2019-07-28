"""Classes that define commission models"""

from abc import ABCMeta, abstractmethod

DEFAULT_MINIMUM_COST_PER_ORDER = 5.0


class AbstractCommissionModel(metaclass=ABCMeta):
    """
    Abstract Commission Model interface.

    Commission models define how much commission should be charged extra to a 
    portfolio per order or trade.
    """

    @abstractmethod
    def calculate(self, order, execution_price):
        """
        Calculate the amount of commission to charge to an :class:``Order`` as 
        the result of a :class:``Trade``.

        :param order: the :py:class:`~order.Order` to charge the commission to.
        :type order: Order
        :param float execution_price: The cost per share.
        :return: amount to charge
        :rtype: float or None
        """

        raise (NotImplementedError("calculate must be overridden"))


class PerOrderCommissionModel(AbstractCommissionModel):
    """
    Calculates commission for a :py:class:`~order.Trade` on a per order basis, 
    so if there is multiple `trade`s created from one `order` commission 
    will only be charged once.
    """

    def __init__(self, cost=DEFAULT_MINIMUM_COST_PER_ORDER):
        """
        :param float cost: The flat cost charged per order.
        """

        self.cost = float(cost)

    def calculate(self, order, execution_price):
        """
        If the order hasn't paid any commission then pay the fixed commission.
        """

        if order.commission == 0.0:
            return self.cost
        else:
            return 0

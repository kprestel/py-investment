"""Classes related to slippage and how it affects how orders get processed."""

from abc import ABCMeta, abstractmethod

class AbstractSlippageModel(metaclass=ABCMeta):
    """
    Abstract Base Class that defines the interface for defining a slippage model.

    Child classes are responsible for implementing a :py:func:`process_order` method.
    """

    def __init__(self):
        self._volume_in_tick = 0

    @property
    def volume_in_tick(self):
        """How much volume was traded in a given tick."""
        return self._volume_in_tick

    @abstractmethod
    def process_order(self, tick_data, order):
        """
        Process how an order gets filled.

        :param tick_data: The data for a given tick
        :param Order order: The order being processed
        :return: The trade that came as a result of the order
        :rtype: Trade
        """

        raise NotImplementedError('process_order must be overridden by child classes')


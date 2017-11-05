"""
Collection of functions and classes that can be used anywhere.
"""
import uuid
import logging
from typing import (
    Iterable,
    Any,
)

import collections

from pytech.exceptions import PyInvestmentTypeError

logger = logging.getLogger(__name__)


def make_id():
    return uuid.uuid4().hex


def is_iterable(obj: Any) -> bool:
    return not isinstance(obj, str) and isinstance(obj, collections.Iterable)


def iterable_to_set(iterable):
    """
    Take an iterable and turn it into a set to ensure that there are no
    duplicate entries.

    :param iterable: Any iterable object who's contents is hashable.
    :return: A set
    :rtype: set
    """
    try:
        return set(iterable)
    except TypeError:
        raise PyInvestmentTypeError(
            'iterable must be an iterable with hashable '
            f'contents! {iterable} is not an iterable!')


def tail(n: int, iterable: Iterable):
    """Return an iterator over the last `n` items"""
    return iter(collections.deque(iterable, maxlen=n))


class Borg(object):
    """A mixin class to make an object act like a singleton"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


# noinspection PyPep8Naming
class class_property(property):
    """
    This is intended to be used a decorator on a classmethod to create a
    class property.

    .. note::

        This property does **NOT** provide a setter.

    >>> class ClassProp(object):
    >>>     _foo = 'bar'
    >>>     @class_property
    >>>     @classmethod
    >>>     def foo(cls):
    >>>         return cls._foo

    """

    # noinspection PyMethodOverriding,PyUnresolvedReferences
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

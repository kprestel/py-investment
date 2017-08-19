"""
Collection of functions and classes that can be used anywhere.
"""
import uuid
import logging
from typing import Iterable

import collections

logger = logging.getLogger(__name__)


def make_id():
    return uuid.uuid4().hex


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
        logger.exception('iterable must be an iterable with hashable '
                         'contents!')
        raise


def tail(n: int, iterable: Iterable):
    """Return an iterator over the last `n` items"""
    return iter(collections.deque(iterable, maxlen=n))


class Borg(object):
    """A mixin class to make an object act like a singleton"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state

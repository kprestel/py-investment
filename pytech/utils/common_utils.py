import uuid
import logging

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
    except TypeError as e:
        logger.exception('iterable must be an iterable with hashable contents!')
        raise e

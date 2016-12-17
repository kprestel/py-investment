"""
Hold exceptions used throughout the package
"""

class PyInvestmentError(Exception):
    """Base exception class for all PyInvestment exceptions"""
    pass

class AssetExistsError(PyInvestmentError):
    """
    Raised when a :class:``Asset`` is trying to be inserted into either :class:``AssetUniverse`` or :class:``Portfolio``
    and already is in the table.  In the event this exception is raised the asset should be updated to whatever the new
    attributes are.
    """
    pass

class AssetNotInUniverseError(PyInvestmentError):
    """
    Raised when an :class:``Asset`` that is not the the :class:``AssetUniverse`` is traded.
    """
    pass

class InvalidPositionError(PyInvestmentError):
    """Raised when a position is not either long or short"""
    pass


class InvalidActionError(PyInvestmentError):
    """Raised when a :class:``Trade`` action is not either buy or sell"""
    pass



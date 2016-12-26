"""
Hold exceptions used throughout the package
"""

class PyInvestmentError(Exception):
    """Base exception class for all PyInvestment exceptions"""
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

class AssetExistsError(PyInvestmentError):
    """
    Raised when a :class:``Asset`` is trying to be inserted into either :class:``AssetUniverse`` or :class:``Portfolio``
    and already is in the table.  In the event this exception is raised the asset should be updated to whatever the new
    attributes are.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

class AssetNotInUniverseError(PyInvestmentError):
    """
    Raised when an :class:``Asset`` that is not the the :class:``AssetUniverse`` is traded.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

class InvalidPositionError(PyInvestmentError):
    """Raised when a position is not either long or short"""
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class InvalidActionError(PyInvestmentError):
    """Raised when a :class:``Trade`` action is not either buy or sell"""
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

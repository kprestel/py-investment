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

class NotAnAssetError(PyInvestmentError):
    """Raised when a subclass of :class:``Asset`` is required but another object is provided"""
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class NotAPortfolioError(PyInvestmentError):
    """Raised when a :class:``Portfolio`` is required but another object is provided"""
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


class InvalidOrderStatusError(PyInvestmentError):
    """Raised when an order status is not valid"""
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class InvalidOrderTypeError(PyInvestmentError):
    """Raised when an order type is not valid"""
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

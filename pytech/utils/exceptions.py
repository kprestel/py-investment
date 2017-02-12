"""
Hold exceptions used throughout the package
"""

class PyInvestmentError(Exception):
    """Base exception class for all PyInvestment exceptions"""

    msg = None

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        # super().__init__(self, *args, **kwargs)

    def message(self):
        return str(self)

    def __str__(self):
        msg = self.msg.format(**self.kwargs)
        return msg

    __unicode__ = __str__
    __repr__ = __str__


class AssetExistsError(PyInvestmentError):
    """
    Raised when a :class:``Asset`` is trying to be inserted into either :class:``AssetUniverse`` or :class:``Portfolio``
    and already is in the table.  In the event this exception is raised the asset should be updated to whatever the new
    attributes are.
    """

    msg = 'Asset with ticker: {ticker} already exists.'

class AssetNotInUniverseError(PyInvestmentError):
    """
    Raised when an :class:``Asset`` that is not the the :class:``AssetUniverse`` is traded.
    """

    msg = 'Could not locate an asset with the ticker: {ticker}.'

class NotAnAssetError(PyInvestmentError):
    """Raised when a subclass of :class:``Asset`` is required but another object is provided"""

    msg = 'asset must be an instance of a subclass of the Asset class. {asset} was provided.'


class NotAPortfolioError(PyInvestmentError):
    """Raised when a :class:``Portfolio`` is required but another object is provided"""

    msg = 'portfolio must be an instance of the portfolio class. {portfolio} was provided.'


class InvalidPositionError(PyInvestmentError):
    """Raised when a position is not either long or short"""

    msg = 'action must either be "BUY" or "SELL". {position} was provided.'
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class InvalidActionError(PyInvestmentError):
    """Raised when a :class:``Trade`` action is not either buy or sell"""

    msg = 'action must either be "BUY" or "SELL". {action} was provided'


class InvalidOrderStatusError(PyInvestmentError):
    """Raised when an order status is not valid"""

    msg = 'order_status must either be "OPEN", "FILLED", "CANCELLED", "REJECTED", or "HELD". {order_status} was provided.'


class InvalidOrderTypeError(PyInvestmentError):
    """Raised when an order type is not valid"""

    msg = 'order_type must either be "STOP", "LIMIT", "STOP_LIMIT", or "MARKET". {order_type} was provided.'


class InvalidOrderTypeParameters(PyInvestmentError):
    """Raised when an :class:``pytech.order.Order`` constructor args are not logically correct."""

    msg = ''

class InvalidOrderSubTypeError(PyInvestmentError):
    """Raised when an order subtype is not valid"""

    msg = 'order_subtype must either be "ALL_OR_NONE", "GOOD_TIL_CANCELED", or "DAY". {order_subtype} was provided.'


class UntriggeredTradeError(PyInvestmentError):
    """Raised when a :class:``pytech.order.Trade`` is made from an order that has not been triggered"""

    msg = 'The order being traded has not been triggered yet. order_id: {id}'


class NotABlotterError(PyInvestmentError):
    """Raised when a :py:class:`pytech.blotter.Blotter` is expected but a different type is provided"""

    msg = 'blotter must be an instance of Blotter. {blotter} was provided.'

class NotAFinderError(PyInvestmentError):

    msg = 'finder must be an instance of Finder. {finder} was provided.'

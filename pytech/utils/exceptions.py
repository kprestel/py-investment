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
    Raised when a :class:``Asset`` is trying to be inserted into either 
    :class:``AssetUniverse`` or :class:``Portfolio`` 
    and already is in the table. In the event this exception is raised the 
    ticker should be updated to whatever the new attributes are.
    """
    msg = 'Asset with ticker: {ticker} already exists.'


class AssetNotInUniverseError(PyInvestmentError):
    """
    Raised when an :class:``Asset`` that is not the 
    the :class:``AssetUniverse`` is traded.
    """
    msg = 'Could not locate an ticker with the ticker: {ticker}.'


class NotAnAssetError(TypeError, PyInvestmentError):
    """
    Raised when a subclass of :class:``Asset`` is required 
    but another object is provided
    """
    msg = ('ticker must be an instance of a subclass of the Asset class. '
           '{ticker} was provided.')


class NotAPortfolioError(TypeError, PyInvestmentError):
    """
    Raised when a :class:``Portfolio`` is required 
    but another object is provided
    """
    msg = ('portfolio must be an instance of the portfolio class. '
           '{portfolio} was provided.')


class InsufficientFundsError(PyInvestmentError):
    """
    Raised when a trade is attempted but there is not sufficient funds to 
    execute the trade.
    """
    msg = 'Insufficient funds to execute trade for ticker: {ticker}'


class InvalidPositionError(ValueError, PyInvestmentError):
    """Raised when a position is not either long or short"""
    msg = 'action must either be "LONG" or "SHORT". {position} was provided.'

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class InvalidActionError(ValueError, PyInvestmentError):
    """Raised when a :class:``Trade`` action is not either buy or sell"""
    msg = 'action must either be "BUY" or "SELL". {action} was provided'


class InvalidOrderStatusError(ValueError, PyInvestmentError):
    """Raised when an order status is not valid"""
    msg = ('order_status must either be "OPEN", "FILLED", "CANCELLED", '
           '"REJECTED", or "HELD". {order_status} was provided.')


class InvalidOrderTypeError(ValueError, PyInvestmentError):
    """Raised when an order type is not valid"""

    msg = ('order_type must either be "STOP", "LIMIT", "STOP_LIMIT", '
           'or "MARKET". {order_type} was provided.')


class InvalidOrderTypeParameters(ValueError, PyInvestmentError):
    """
    Raised when an :class:``pytech.order.Order`` 
    constructor args are not logically correct.
    """
    msg = ''


class InvalidOrderSubTypeError(ValueError, PyInvestmentError):
    """Raised when an order subtype is not valid"""
    msg = ('order_subtype must either be "ALL_OR_NONE", "GOOD_TIL_CANCELED", '
           'or "DAY". {order_subtype} was provided.')


class UntriggeredTradeError(PyInvestmentError):
    """
    Raised when a :class:``pytech.order.Trade`` is made from an order 
    that has not been triggered
    """
    msg = 'The order being traded has not been triggered yet. order: {order}'


class NotABlotterError(TypeError, PyInvestmentError):
    """
    Raised when a :py:class:`pytech.blot.Blotter` is expected but a different 
    type is provided
    """
    msg = 'blot must be an instance of Blotter. {blot} was provided.'


class NotAFinderError(TypeError, PyInvestmentError):
    msg = 'finder must be an instance of Finder. {finder} was provided.'


class InvalidEventTypeError(TypeError, PyInvestmentError):
    msg = ('Invalid EventType. Must be {expected}. '
           '{event_type} was provided.')


class InvalidSignalTypeError(TypeError, PyInvestmentError):
    """Raised when a signal type is not right."""
    msg = ('Invalid SignalType. Must be in the SignalType enum. '
           '{signal_type} was provided.')


class BadOrderParams(TypeError, PyInvestmentError):
    """Raised when an order is placed that is illegal."""
    msg = 'Attempted to place an order with a {order_type} of {price}'


class TradeControlViolation(PyInvestmentError):
    msg = ('Order for {qty} shares of {ticker} at {dt} violates trading '
           'constraint {constraint}')

"""
Hold exceptions used throughout the package
"""

class AssetExistsException(Exception):
    """
    Raised when a :class:``Asset`` is trying to be inserted into either :class:``AssetUniverse`` or :class:``Portfolio``
    and already is in the table.  In the event this exception is raised the asset should be updated to whatever the new
    attributes are.
    """
    pass

class AssetNotInUniverseException(Exception):
    """
    Raised when an :class:``Asset`` that is not the the :class:``AssetUniverse`` is traded.
    """
    pass

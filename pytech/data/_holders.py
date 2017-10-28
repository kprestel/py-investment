"""
This module is for lightweight data holders to make interfacing the
return values these functions easier.
"""
from collections import namedtuple
from typing import TYPE_CHECKING

import pandas as pd


class ReaderResult(object):
    """Data holder object that gets returned by readers."""

    def __init__(self, lib_name: str, ticker: str, df: pd.DataFrame=None,
                 successful: bool=True) -> None:
        """
        Constructor for the result.

        :param df: the ``DataFrame``.
        :param lib_name: the library name that it came from.
        :param ticker: the ticker that the result is associated with.
        :param successful: defaults to ``True``. Used to indicate whether or
            not there was an error reading the data.
        """
        # have to explicitly check if its None, other wise pandas throws.
        if df is None:
            self.df = pd.DataFrame()
        else:
            self.df = df
        self.lib_name = lib_name
        self.ticker = ticker
        self.successful = successful

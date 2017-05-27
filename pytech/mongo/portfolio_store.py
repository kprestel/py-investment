import datetime as dt
import logging
from typing import Any, Dict, Union

import pandas as pd
from arctic.date import DateRange
from arctic.decorators import mongo_retry
from arctic.exceptions import DuplicateSnapshotException
from arctic.store.version_store import VersionStore
from arctic.store.versioned_item import VersionedItem


class PortfolioStore(VersionStore):
    """
    Wrapper for the :class:``arctic.store.version_store.VersionStore`` to
    persist portfolio data.
    """

    LIBRARY_TYPE = 'pytech.Portfolio'
    LIBRARY_NAME = 'pytech.portfolio'

    def __init__(self, arctic_lib):
        self.logger = logging.getLogger(__name__)
        super().__init__(arctic_lib)
        self.logger.info('PortfolioStore collection name: '
                         f'{arctic_lib.get_name()}')

    @mongo_retry
    def read(self, symbol: str,
             as_of: str or int or dt.datetime = None,
             date_range: DateRange = None,
             from_version: Any = None,
             allow_secondary: bool or None = None,
             return_metadata: bool = False,
             **kwargs) -> Union[VersionedItem, pd.DataFrame, pd.Series]:
        """
        Read data for the named symbol. Returns a ``VersionedItem`` object
        with a data and metadata element that were passed to the ``write``
        function.

        :param symbol: Name for the item.
        :param as_of:

            * int: specific version number
            * str: snapshot name
            * datetime the version that existed at that point in time.

        :param date_range: ``DateRange`` to read data for.
        :param from_version:
        :param allow_secondary:
        :param return_metadata: If true then a ``VersionedItem`` will be
        returned, otherwise only the data will be returned.
        :param kwargs:
        :return: A ``VersionedItem`` or a pandas DataFrame or Series.
        """
        versioned_item = super().read(symbol, as_of, date_range, from_version,
                                      allow_secondary, **kwargs)
        if return_metadata:
            return versioned_item
        else:
            return versioned_item.data

    def write_snapshot(self, symbol: str,
                       data: pd.Series or pd.DataFrame,
                       snap_shot: dt.datetime or str,
                       metadata: Dict = None,
                       prune_previous_version: bool = False,
                       **kwargs) -> VersionedItem:
        """
        Write the data for the named symbol and also create a snapshot with
        the given date.

        :param symbol:
        :param snap_shot:
        :param data:
        :param metadata:
        :param prune_previous_version:
        :param kwargs:
        :return:
        """
        versioned_item = super().write(symbol, data, metadata,
                                       prune_previous_version, **kwargs)
        try:
            self.logger.info(f'Writing snapshot with name: {snap_shot}')
            super().snapshot(snap_shot)
        except DuplicateSnapshotException:
            self.logger.debug('Snapshot with name: '
                              f'{snap_shot} already exists.')
        return versioned_item

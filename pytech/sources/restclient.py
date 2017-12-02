import logging
import pandas as pd
from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import (
    Union,
    Optional,
)

import requests
from requests.exceptions import HTTPError

from exceptions import DataAccessError
from utils import DateRange
from utils.enums import AutoNumber


class RestClientError(DataAccessError):
    pass


class HTTPAction(AutoNumber):
    GET = ()
    POST = ()
    PUT = ()
    DELETE = ()

    @classmethod
    def check_if_valid(cls, value):
        name = super().check_if_valid(value)
        if name is not None:
            return name
        else:
            raise RestClientError('Invalid HTTP Action')


class RestClient(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs) -> None:
        self.logger = logging.getLogger(__name__)

        if kwargs.get('session'):
            self._session = requests.Session()
        else:
            self._session = requests

    def __repr__(self):
        return f'{self.__class__.__name__}(base_url={self.base_url})'

    @property
    @abstractmethod
    def headers(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def base_url(self):
        raise NotImplementedError

    @abstractmethod
    def get_intra_day(self, ticker: str,
                      date_range: DateRange,
                      freq: str = '5min',
                      **kwargs) -> pd.DataFrame:
        """
        Get intra day trade data. How far back data is available will vary from
        source to source.

        .. note::
            Depending on the data source and the range of data requested, it
            is not guaranteed that all data will be available.

        :param ticker: The ticker to retrieve intra data for.
        :param date_range: The range of dates to get data for.
        :param freq: The frequency of the ticks.
        :return: A :class:`pd.DataFrame` with the data.
        """
        raise NotImplementedError

    @abstractmethod
    def get_historical_data(self, ticker: str,
                            date_range: DateRange,
                            freq: str = 'Daily',
                            adjusted: bool = True):
        raise NotImplementedError

    def _request(self, url: Optional[str],
                 method: Union[HTTPAction, str] = HTTPAction.GET,
                 **kwargs):
        """
        Make the HTTP request and return the response.

        :param method: the HTTP method. Must be:

            - GET
            - POST
            - PUT
            - DELETE

        :param url: the url to make the request to. This is appended to
            `base_url`.
        :param kwargs: passed directly to the :class:`requests.requests`
            object.
        :return: the response.
        """
        method = HTTPAction.check_if_valid(method).name

        if url is None:
            url = self.base_url
        else:
            url = f'{self.base_url}/{url}'

        resp = self._session.request(method, url, headers=self.headers,
                                     **kwargs)

        try:
            resp.raise_for_status()
        except HTTPError as e:
            self.logger.exception(resp.content)
            raise RestClientError(e)
        else:
            return resp

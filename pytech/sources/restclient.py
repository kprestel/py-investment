import logging
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

from utils.enums import AutoNumber


class RestClientError(Exception):
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

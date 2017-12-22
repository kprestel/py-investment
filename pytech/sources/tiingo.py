import json
import os

import pandas as pd

import pytech.utils as utils
from .restclient import (
    RestClient,
    RestClientError,
)
from pytech.utils import (
    DateRange,
    Dict,
)


class TiingoClient(RestClient):
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self._base_url = 'https://api.tiingo.com'
        self.api_key = os.environ.get('TIINGO_API_KEY', api_key)

        if self.api_key is None:
            raise KeyError('Must set TIINGO_API_KEY.')

        self._headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'pytech-client'
        }

    @property
    def base_url(self):
        return self._base_url

    @property
    def headers(self):
        return self._headers

    def _get_dt_params(self, date_range: DateRange):
        params = {}

        if date_range is None:
            return params

        if date_range.start is not None:
            params['startDate'] = date_range.start.strftime('%Y-%m-%d')

        if date_range.end is not None:
            params['endDate'] = date_range.end.strftime('%Y-%m-%d')

        return params

    def get_ticker_metadata(self, ticker: str) -> Dict[str, str]:
        """
        Returns metadata for a single ticker.
        :param ticker: the ticker for the asset.
        :return: a :class:`pd.DataFrame` with the response.
        """
        resp = self._request(f'/tiingo/daily/{ticker}')
        return resp.json()

    def get_historical_data(self, ticker: str,
                            date_range: DateRange = None,
                            freq: str = 'daily',
                            adjusted: bool = True,
                            persist: bool = True,
                            **kwargs) -> pd.DataFrame:
        url = f'/tiingo/daily/{ticker}/prices'
        params = {
            'format': 'json',
            'resampleFreq': freq.lower(),
        }

        params.update(self._get_dt_params(date_range))

        resp = self._request(url=url, params=params)

        df = pd.read_json(json.dumps(resp.json()))

        if df.empty:
            raise RestClientError('Empty DataFrame was returned')

        df = utils.clean_df(df, ticker)

        if persist:
            self._persist_df(df)

        return df

    def get_intra_day(self, ticker: str,
                      date_range: DateRange = None,
                      freq: str = '5min',
                      persist: bool = True,
                      **kwargs):
        url = f'/iex/{ticker}/prices'
        params = {
            'ticker': ticker,
            'resampleFreq': freq
        }
        params.update(self._get_dt_params(date_range))

        resp = self._request(url, params=params)
        df = pd.read_json(json.dumps(resp.json()))

        if df.empty:
            raise RestClientError('Empty DataFrame was returned')

        df = utils.clean_df(df, ticker)

        if persist:
            self._persist_df(df)

        return df

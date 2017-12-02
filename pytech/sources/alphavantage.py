import os
from io import BytesIO
from typing import (
    Dict,
    Optional,
    Union,
)

import pandas as pd

import pytech.utils as utils
from pytech.exceptions import PyInvestmentValueError
from sources.restclient import (
    HTTPAction,
    RestClient,
    RestClientError,
)


class AlphaVantageClient(RestClient):
    _valid_intervals = {'1min', '5min', '15min', '30min', '60min'}
    _valid_outputsizes = {'compact', 'full'}

    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self._base_url = 'https://www.alphavantage.co/query'
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', api_key)

        if self.api_key is None:
            raise KeyError('Must set ALPHA_VANTAGE_API_KEY')

        self._headers = {
            'User-Agent': 'pytech-client'
        }

    @property
    def base_url(self):
        return self._base_url

    @property
    def headers(self):
        return self._headers

    def _get_base_ts_params(self, ticker: str,
                            ts_func: str,
                            outputsize: str) -> Dict[str, str]:
        return {
            'symbol': ticker,
            'outputsize': outputsize,
            'apikey': self.api_key,
            'function': f'TIME_SERIES_{ts_func}',
            'datatype': 'csv'
        }

    def _request(self, url: Optional[str],
                 method: Union[HTTPAction, str] = HTTPAction.GET,
                 **kwargs) -> pd.DataFrame:
        resp = super()._request(url, method, stream=True, **kwargs)
        df = pd.read_csv(BytesIO(resp.content), encoding='utf8')
        if df.empty:
            raise RestClientError('Empty DataFrame was returned')

        df = utils.rename_bar_cols(df)
        df[utils.DATE_COL] = df[utils.DATE_COL].apply(utils.parse_date)
        return df

    def get_intra_day(self, ticker: str,
                      interval: str = '15min',
                      outputsize='compact') -> pd.DataFrame:
        if interval not in self._valid_intervals:
            raise PyInvestmentValueError(f'{interval} is not a valid interval')

        if outputsize not in self._valid_outputsizes:
            raise PyInvestmentValueError(f'{outputsize} is not a valid '
                                         f'outputsize')

        params = self._get_base_ts_params(ticker, 'INTRADAY', outputsize)
        params['interval'] = interval
        return self._request(None, params=params)

    def get_daily(self, ticker: str,
                  outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'DAILY', outputsize)
        df = self._request(None, params=params)
        return df

    def get_daily_adj(self, ticker: str,
                      outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'DAILY_ADJUSTED', outputsize)
        return self._request(None, params=params)

    def get_weekly(self, ticker: str,
                   outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'WEEKLY', outputsize)
        return self._request(None, params=params)

    def get_weekly_adj(self, ticker: str,
                       outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'WEEKLY_ADJUSTED',
                                          outputsize)
        return self._request(None, params=params)

    def get_monthly(self, ticker: str,
                    outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'MONTHLY', outputsize)
        return self._request(None, params=params)

    def get_monthly_adj(self, ticker: str,
                        outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'MONTHLY_ADJUSTED',
                                          outputsize)
        return self._request(None, params=params)

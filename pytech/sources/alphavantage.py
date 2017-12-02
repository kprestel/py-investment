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
from utils import DateRange


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

    def _request(self, url: Optional[str] = None,
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
                      date_range: DateRange,
                      freq: str = '5min',
                      **kwargs) -> pd.DataFrame:
        """
        Get intra day trade data and return it in a :class:`pd.DataFrame`.

        Typically there is 10 - 15 past trading days worth of data available
        from `AlphaVantage`.

        If the `date_range` + `freq` combination results in more than 100
        ticks of data being requested then **all** available data will be
        requested.

        :param ticker: The ticker to retrieve intra data for.
        :param date_range: The range of dates to get data for.
        :param freq: The interval of the ticks.

            Valid options are:

                * 1min
                * 5min
                * 15min
                * 30min
                * 60min

        :keyword drop_extra: If `True` then any data outside of the requested
            `date_range` will be dropped before being returned.
            Defaults to `False`

        :return: The :class:`pd.DataFrame` with the data.
        """
        prng = pd.period_range(date_range.start, date_range.end,
                               freq=pd.Timedelta(freq))
        if prng.size > 100:
            outputsize = 'full'
        else:
            outputsize = 'compact'

        params = self._get_base_ts_params(ticker, 'INTRADAY', outputsize)
        params['interval'] = freq
        df = self._request(params=params)

        drop_extra = kwargs.get('drop_extra', False)

        if drop_extra:
            df = df[date_range.start:date_range.end]

        return df

    def get_historical_data(self, ticker: str,
                            date_range: DateRange,
                            freq: str = 'Daily',
                            adjusted: bool = True):
        pass

    def get_daily(self, ticker: str,
                  outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'DAILY', outputsize)
        df = self._request(params=params)
        return df

    def get_daily_adj(self, ticker: str,
                      outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'DAILY_ADJUSTED', outputsize)
        return self._request( params=params)

    def get_weekly(self, ticker: str,
                   outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'WEEKLY', outputsize)
        return self._request(params=params)

    def get_weekly_adj(self, ticker: str,
                       outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'WEEKLY_ADJUSTED',
                                          outputsize)
        return self._request(params=params)

    def get_monthly(self, ticker: str,
                    outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'MONTHLY', outputsize)
        return self._request(params=params)

    def get_monthly_adj(self, ticker: str,
                        outputsize: str = 'compact') -> pd.DataFrame:
        params = self._get_base_ts_params(ticker, 'MONTHLY_ADJUSTED',
                                          outputsize)
        return self._request(params=params)

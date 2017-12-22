import os
from io import BytesIO
from typing import (
    Dict,
    Optional,
    Union,
)

import pandas as pd

import pytech.utils as utils
from .restclient import (
    HTTPAction,
    RestClient,
    RestClientError,
)
from pytech.utils import DateRange


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

        return df

    def _get_outputsize(self, date_range: DateRange, freq: str):
        """
        Returns the outputsize based on whether the `freq` + `date_range`
        combination results in a period of greater than 100 ticks.
        """
        prng = pd.period_range(date_range.start, date_range.end,
                               freq=pd.Timedelta(int(freq[0]), freq[1]))
        if prng.size > 100:
            return 'full'
        else:
            return 'compact'

    def get_intra_day(self, ticker: str,
                      date_range: DateRange,
                      freq: str = '5min',
                      persist: bool = True,
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

        :param persist: If `True` write the data to the database. This will
            occur before any data has been dropped from the `df`.
        :keyword drop_extra: If `True` then any data outside of the requested
            `date_range` will be dropped before being returned.
            Defaults to `False`
        :keyword outputsize: Set the output size.
            * full
                * all available data
            * compact
                * last 100 records
        :return: The :class:`pd.DataFrame` with the data.
        """
        if 'outputsize' in kwargs:
            outputsize = kwargs.pop('outputsize')
        else:
            outputsize =  self._get_outputsize(date_range, freq)

        params = self._get_base_ts_params(ticker, 'INTRADAY', outputsize)
        params['interval'] = freq
        df = self._request(params=params)
        df = utils.clean_df(df, ticker)

        if persist:
            self._persist_df(df)

        if kwargs.get('drop_extra', False):
            df = df[date_range.start:date_range.end]

        return df

    def get_historical_data(self, ticker: str,
                            date_range: DateRange,
                            freq: str = 'Daily',
                            adjusted: bool = True,
                            persist: bool = True,
                            **kwargs) -> pd.DataFrame:
        ts_param = freq.upper() + '_ADJUSTED' if adjusted else freq.upper()
        if 'DAILY' in ts_param:
            outputsize = self._get_outputsize(date_range, '1D')
        elif 'WEEKLY' in ts_param:
            outputsize = self._get_outputsize(date_range, '1W')
        elif 'MONTHLY' in ts_param:
            outputsize = self._get_outputsize(date_range, '1M')
        else:
            raise ValueError(f'{freq} is not a valid frequency')

        params = self._get_base_ts_params(ticker, ts_param, outputsize)
        df = self._request(params=params)
        df = utils.clean_df(df, ticker, adjusted=adjusted)

        if persist:
            self._persist_df(df)

        if kwargs.get('drop_extra', False):
            df = df[date_range.start:date_range.end]

        return df

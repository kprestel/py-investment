# noinspection PyUnresolvedReferences
import time

import pandas as pd
from sqlalchemy import select

from pytech.data.connection import (
    reader,
    write,
)
from pytech.data.schema import (
    assets,
    portfolio,
)


class TestWrite(object):
    writer = write()

    def test___call__(self, basic_portfolio):
        self.writer.insert_portfolio(basic_portfolio)
        # ins = portfolio.insert().values(cash=123456)
        # self.writer(ins)

    def test_df(self, aapl_df: pd.DataFrame):
        ins = assets.insert().values(ticker='AAPL')
        self.writer(ins)
        aapl_df['ticker'] = 'AAPL'
        self.writer.df(aapl_df, 'bar')


class TestRead(object):
    reader_ = reader()
    q = select([portfolio])
    for row in reader_(q):
        assert row is not None

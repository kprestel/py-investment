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

    def test___call__(self):
        ins = portfolio.insert().values(cash=123456)
        self.writer(ins)

    def test_df(self, goog_df: pd.DataFrame):
        ins = assets.insert().values(ticker='GOOG')
        self.writer(ins)
        goog_df['ticker'] = 'GOOG'
        self.writer.df(goog_df, 'bar')


class TestRead(object):
    reader_ = reader()
    q = select([portfolio])
    for row in reader_(q):
        assert row is not None

# noinspection PyUnresolvedReferences
import time

import pandas as pd
import pytest
from sqlalchemy import select

from pytech.data.connection import (
    reader,
    write,
)
from pytech.data.schema import (
    assets,
    portfolio,
)


@pytest.mark.skip('todo?')
class TestWrite(object):
    writer = write()

    def test___call__(self, basic_portfolio):
        self.writer.insert_portfolio(basic_portfolio)
        # ins = portfolio.insert().values(cash=123456)
        # self.writer(ins)

    def test_df(self, fb_df: pd.DataFrame):
        ins = assets.insert().values(ticker='FB')
        self.writer(ins)
        fb_df['ticker'] = 'FB'
        self.writer.df(fb_df, 'bar')


class TestRead(object):
    reader_ = reader()
    q = select([portfolio])
    for row in reader_(q):
        assert row is not None

# noinspection PyUnresolvedReferences
import time

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

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

    def test___call__none(self, basic_portfolio):
        self.writer.insert_portfolio(basic_portfolio, on_conflict=None)

    def test___call__raise(self, basic_portfolio):
        with pytest.raises(IntegrityError):
            self.writer.insert_portfolio(basic_portfolio, on_conflict='raise')


    @pytest.mark.skip('todo?')
    def test_df(self, fb_df: pd.DataFrame):
        ins = assets.insert().values(ticker='FB')
        self.writer(ins)
        fb_df['ticker'] = 'FB'
        self.writer.df(fb_df, 'bar')


class TestRead(object):
    def test_read(self):
        reader_ = reader()
        q = select([portfolio])
        for row in reader_(q):
            assert row is not None

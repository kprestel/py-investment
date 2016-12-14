from contextlib import contextmanager
from distutils import dirname
from os.path import join

import pytest
import pandas as pd
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy import event
# from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from pytech.base import Base
from pytech.portfolio import Portfolio
from pytech.stock import Stock, Fundamental

import logging

PROJECT_DIR = dirname(__file__)
DATABASE_LOCATION = join(PROJECT_DIR, 'pytech_test.sqlite')
cs = 'sqlite+pysqlite:///{}'.format(DATABASE_LOCATION)
engine = create_engine(cs, connect_args={'check_same_thread': False}, poolclass=StaticPool)
Session = sessionmaker(bind=engine)


@pytest.yield_fixture(scope='session', autouse=True)
def tables():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.yield_fixture(autouse=True)
def mock_session(monkeypatch):
    @contextmanager
    def session(auto_close=True):
        session = Session()
        yield session
        session.commit()
        if auto_close:
            session.close()

    monkeypatch.setattr('pytech.db_utils.transactional_session', session)


@pytest.yield_fixture(autouse=True)
def mock_query_session(monkeypatch):
    @contextmanager
    def session():
        session = Session()
        yield session
        session.close()

    monkeypatch.setattr('pytech.db_utils.query_session', session)


@pytest.fixture(autouse=True)
def mock_fundamentals(monkeypatch):
    aapl_fundamentals = {
        'accrued_liabilities_current': 22027000000.0,
        'acts_pay_current': 37294000000.0,
        'acts_receive_current': 15754000000.0,
        'amend': False,
        'owned_assets': 321686000000.0,
        'cash': 20484000000.0,
        'fin_cash_flow': -20483000000.0,
        'inv_cash_flow': -45977000000.0,
        'ops_cash_flow': 65824000000.0,
        'cogs': 131376000000.0,
        'common_stock_outstanding': 5336166000.0,
        'comprehensive_income_net_of_tax': 46666000000.0,
        'current_assets': 106869000000.0,
        'current_liabilities': 79006000000.0,
        'depreciation_amortization': 10505000000.0,
        'dividend': 2.18,
        'doc_type': '10-K',
        'end_date': '2016-09-24',
        'eps': 8.35,
        'eps_diluted': 8.31,
        'equity': 128249000000.0,
        'year': 2016,
        'gross_profit': 84263000000.0,
        'interest_expense': 1456000000.0,
        'inventory_net': 2132000000.0,
        'net_income': 45687000000.0,
        'net_taxes_paid': 10444000000.0,
        'op_income': 60024000000.0,
        'period_focus': 'FY',
        'property_plant_equipment': 61245000000.0,
        'research_and_dev_expense': 10045000000.0,
        'revenues': 215639000000.0,
        'shares_outstanding': 5470820000.0,
        'ticker': 'AAPL',
        'tax_expense': 15685000000.0,
        'total_liabilities': 193437000000.0,
        'warranty_accrual': 3702000000.0,
        'warranty_accrual_payments': 4663000000.0
    }

    msft_fundamentals = {
        'acts_pay_current': 6296000000.0,
        'acts_receive_current': 11129000000.0,
        'amend': False,
        'owned_assets': 212524000000.0,
        'cash': 13928000000.0,
        'fin_cash_flow': 14329000000.0,
        'inv_cash_flow': -18470000000.0,
        'ops_cash_flow': 11549000000.0,
        'common_stock_outstanding': 7784000000.0,
        'comprehensive_income_net_of_tax': 4834000000.0,
        'cur_assets': 157909000000.0,
        'cur_liab': 58810000000.0,
        'dividend': 0.39,
        'doc_type': '10-Q',
        'end_date': '2016-09-30',
        'eps': 0.6,
        'eps_diluted': 0.6,
        'equity': 70372000000.0,
        'year': 2017,
        'gross_profit': 12609000000.0,
        'interest_expense': 437000000.0,
        'inventory_net': 3122000000.0,
        'net_income': 4690000000.0,
        'operating_income': 5225000000.0,
        'period_focus': 'Q1',
        'research_and_dev_expense': 3106000000.0,
        'revenues': 20453000000.0,
        'shares_outstanding': 7789000000.0,
        'ticker': 'MSFT',
        'tax_expense': 635000000.0,
        'total_liabilities': 142152000000.0
    }

    def mock_spiders():
        with test_session() as session:
            session.add(Fundamental.from_dict(fundamental_dict=aapl_fundamentals))


@contextmanager
def test_session():
    session = Session()
    yield session
    session.commit()
    session.close()


@pytest.fixture(scope='class')
def stock_universe():
    """Write stocks to the DB"""
    Stock.from_list(ticker_list=['AAPL'], start='20160901', end='20161201', get_fundamentals=True,
                    get_ohlcv=True)


class TestStock(object):
    def test_stock_start_date_constructor(self):
        with pytest.raises(ValueError):
            stock = Stock(ticker='AAPL', start_date='not a date', end_date='20161124')

    def test_stock_end_date_constructor(self):
        with pytest.raises(ValueError):
            stock = Stock(ticker='AAPL', start_date='20161124', end_date='not a date')

    def test_stock_constructor_date_logic_check(self):
        """
        the start date should not be more recent than the end date
        """
        with pytest.raises(ValueError):
            stock = Stock(ticker='AAPL', start_date='20161124', end_date='20161123')

    def test_stock_constructor(self):
        """
        testing the default constructor
        """
        stock = Stock(ticker='AAPL', start_date='20151124', end_date='20161124')
        test_start_date = datetime(year=2015, month=11, day=24)
        test_end_date = datetime(year=2016, month=11, day=24)
        assert stock.ticker == 'AAPL'
        assert stock.start_date == test_start_date
        assert stock.end_date == test_end_date
        assert stock.get_ohlcv == True
        assert stock.fundamentals == {}

    def test_stock_fixture(self, stock: Stock):
        assert stock.ticker == 'AAPL'
        assert stock.start_date == datetime(year=2016, month=9, day=1)
        assert stock.end_date == datetime(year=2016, month=11, day=1)
        assert stock.get_ohlcv == True
        assert stock.fundamentals == {}

    @pytest.mark.skip(reason='fixture is dumb')
    def test_stock_with_fundamentals_fixture(self, stock_with_fundamentals: Stock):
        assert stock_with_fundamentals.ticker == 'AAPL'
        assert stock_with_fundamentals.start_date == datetime(year=2016, month=9, day=1)
        assert stock_with_fundamentals.end_date == datetime(year=2016, month=11, day=1)
        assert stock_with_fundamentals.get_ohlcv == True
        assert len(stock_with_fundamentals.fundamentals) != 0

    def test_stock_with_ohlcv_fixture(self, stock_with_ohlcv: Stock):
        assert stock_with_ohlcv.ticker == 'AAPL'
        assert stock_with_ohlcv.start_date == datetime(year=2016, month=9, day=1)
        assert stock_with_ohlcv.end_date == datetime(year=2016, month=11, day=1)
        assert stock_with_ohlcv.get_ohlcv == True
        assert stock_with_ohlcv.fundamentals == {}
        assert type(stock_with_ohlcv.ohlcv) == pd.DataFrame


@pytest.mark.usefixtures('stock_universe')
class TestPortfolio(object):
    def test_make_trade(self):
        portfolio = Portfolio()
        portfolio.make_trade(ticker='AAPL', qty=100, action='buy')
        for k, owned_stock in portfolio.owned_assets.items():
            assert isinstance(owned_stock.stock, Stock)
            for key, fundamental in owned_stock.stock.fundamentals.items():
                assert isinstance(fundamental, Fundamental)

    def test_update_trade(self):
        portfolio = Portfolio()
        portfolio.make_trade(ticker='AAPL', qty=100, action='buy')
        portfolio.make_trade(ticker='AAPL', qty=100, action='buy')
        aapl = portfolio.owned_assets.get('AAPL')
        assert aapl.shares_owned == 200


@pytest.mark.skip(reason='I will fix it later')
class TestFundamental(object):
    def test_fundamental_fixture(self, fundamental: Fundamental):
        """
        test the fundamental fixture from conftest.py
        """
        assert fundamental.amended == False
        assert fundamental.assets == 10000.0
        assert fundamental.current_assets == 5000.0
        assert fundamental.current_liabilities == 5000.0
        assert fundamental.cash == 4000.0
        assert fundamental.dividend == 1.0
        assert fundamental.end_date == datetime(year=2016, month=10, day=1)
        assert fundamental.eps == .5
        assert fundamental.eps_diluted == .4
        assert fundamental.equity == 5000.0
        assert fundamental.net_income == 6000.0
        assert fundamental.operating_income == 5000.0
        assert fundamental.revenues == 10000.0
        assert fundamental.investment_revenues == 4000.0
        assert fundamental.fin_cash_flow == 500.0
        assert fundamental.inv_cash_flow == 400.0
        assert fundamental.ops_cash_flow == 700.0
        assert fundamental.period_focus == 'Q3'
        assert fundamental.year == '2016'

    def test_creating_fundamental_from_dict(self, fundamental: Fundamental):
        fundamental_dict = {
            'amended': False,
            'owned_assets': 10000.0,
            'current_assets': 5000.0,
            'current_liabilities': 5000.0,
            'cash': 4000.0,
            'dividend': 1.0,
            'end_date': '20161001',
            'eps': .5,
            'eps_diluted': .4,
            'equity': 5000.0,
            'net_income': 6000.0,
            'operating_income': 5000.0,
            'revenues': 10000.0,
            'investment_revenues': 4000.0,
            'fin_cash_flow': 500.0,
            'inv_cash_flow': 400.0,
            'ops_cash_flow': 700.0,
            'period_focus': 'Q3',
            'year': '2016',
            'property_plant_equipment': 5000.0,
            'gross_profit': 7000.0,
            'tax_expense': 700.0,
            'net_taxes_paid': 400.0,
            'acts_pay_current': 400.0,
            'acts_receive_current': 600.0,
            'acts_receive_noncurrent': 800.0,
            'accrued_liabilities_current': 500.0
        }

        fundamental_from_dict = Fundamental.from_dict(fundamental_dict=fundamental_dict)
        assert fundamental.amended == fundamental_from_dict.amended
        assert fundamental.assets == fundamental_from_dict.assets
        assert fundamental.current_assets == fundamental_from_dict.current_assets
        assert fundamental.current_liabilities == fundamental_from_dict.current_liabilities
        assert fundamental.cash == fundamental_from_dict.cash
        assert fundamental.dividend == fundamental_from_dict.dividend
        assert fundamental.end_date == fundamental_from_dict.end_date
        assert fundamental.eps == fundamental_from_dict.eps
        assert fundamental.eps_diluted == fundamental_from_dict.eps_diluted
        assert fundamental.equity == fundamental_from_dict.equity
        assert fundamental.net_income == fundamental_from_dict.net_income
        assert fundamental.operating_income == fundamental_from_dict.operating_income
        assert fundamental.revenues == fundamental_from_dict.revenues
        assert fundamental.investment_revenues == fundamental_from_dict.investment_revenues
        assert fundamental.fin_cash_flow == fundamental_from_dict.fin_cash_flow
        assert fundamental.inv_cash_flow == fundamental_from_dict.inv_cash_flow
        assert fundamental.ops_cash_flow == fundamental_from_dict.ops_cash_flow
        assert fundamental.period_focus == fundamental_from_dict.period_focus
        assert fundamental.year == fundamental_from_dict.year
        assert fundamental.ticker == fundamental_from_dict.ticker
        assert fundamental.property_plant_equipment == fundamental_from_dict.property_plant_equipment
        assert fundamental.gross_profit == fundamental_from_dict.gross_profit
        assert fundamental.tax_expense == fundamental_from_dict.tax_expense
        assert fundamental.net_taxes_paid == fundamental_from_dict.net_taxes_paid
        assert fundamental.acts_pay_current == fundamental_from_dict.acts_pay_current
        assert fundamental.acts_receive_current == fundamental_from_dict.acts_receive_current
        assert fundamental.acts_receive_noncurrent == fundamental_from_dict.acts_receive_noncurrent
        assert fundamental.accrued_liabilities_current == fundamental_from_dict.accrued_liabilities_current

    def test_current_ratio(self, fundamental: Fundamental):
        assert fundamental.current_ratio() == 1

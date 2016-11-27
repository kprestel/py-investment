import pytest
import pandas as pd
from datetime import datetime
from pytech.stock import Stock, Fundamental

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
        assert stock.get_ohlcv == False
        assert stock.fundamentals == []

    def test_stock_fixture(self, stock: Stock):
        assert stock.ticker == 'AAPL'
        assert stock.start_date == datetime(year=2016, month=9, day=1)
        assert stock.end_date == datetime(year=2016, month=11, day=1)
        assert stock.get_ohlcv == False
        assert stock.fundamentals == []

    def test_stock_with_fundamentals_fixture(self, stock_with_fundamentals: Stock):
        assert stock_with_fundamentals.ticker == 'AAPL'
        assert stock_with_fundamentals.start_date == datetime(year=2016, month=9, day=1)
        assert stock_with_fundamentals.end_date == datetime(year=2016, month=11, day=1)
        assert stock_with_fundamentals.get_ohlcv == False
        assert len(stock_with_fundamentals.fundamentals) != 0

    def test_stock_with_ohlcv_fixture(self, stock_with_ohlcv: Stock):
        assert stock_with_ohlcv.ticker == 'AAPL'
        assert stock_with_ohlcv.start_date == datetime(year=2016, month=9, day=1)
        assert stock_with_ohlcv.end_date == datetime(year=2016, month=11, day=1)
        assert stock_with_ohlcv.get_ohlcv == True
        assert stock_with_ohlcv.fundamentals == []
        assert type(stock_with_ohlcv.ohlcv) == pd.DataFrame


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
        assert fundamental.ticker == 'AAPL'

    def test_creating_fundamental_from_dict(self, fundamental: Fundamental):
        fundamental_dict = {
            'amended': False,
            'assets': 10000.0,
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
            'ticker': 'AAPL',
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





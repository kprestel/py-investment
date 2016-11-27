import pytest
from pytech.stock import Stock, Fundamental

@pytest.fixture()
def stock():
    return Stock(ticker='AAPL', start_date='20160901', end_date='20161101')

@pytest.fixture()
def stock_with_fundamentals():
    return Stock(ticker='AAPL', start_date='20160901', end_date='20161101', get_fundamentals=True)

@pytest.fixture()
def stock_with_ohlcv():
    return Stock(ticker='AAPL', start_date='20160901', end_date='20161101', get_fundamentals=False, get_ohlcv=True)

@pytest.fixture()
def stock_full_stock():
    return Stock(ticker='AAPL', start_date='20160901', end_date='20161101', get_fundamentals=True, get_ohlcv=True)

@pytest.fixture()
def fundamental():
    return Fundamental(amended=False, assets=10000.0, current_assets=5000.0, current_liabilities=5000.0, cash=4000.0,
                       dividend=1.0, end_date='20161001', eps=.5, eps_diluted=.4, equity=5000.0, net_income=6000.0,
                       operating_income=5000.0, revenues=10000.0, investment_revenues=4000.0, fin_cash_flow=500.0,
                       inv_cash_flow=400.0, ops_cash_flow=700.0, period_focus='Q3', year='2016', ticker='AAPL')



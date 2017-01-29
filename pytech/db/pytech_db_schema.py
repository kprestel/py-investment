from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import Numeric
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

metadata = MetaData()

PYTECH_DB_TABLE_NAMES = frozenset({
    'asset',
    'fundamental',
    'owned_asset',
    'owned_stock',
    'portfolio',
    'trade',
    'order',
    'universe_ohlcv'
})

asset = Table('asset', metadata,
              Column('id', Integer, primary_key=True),
              Column('ticker', String, unique=True, nullable=False, index=True, primary_key=True),
              Column('start_date', Integer, nullable=False),
              Column('end_date', Integer, nullable=False),
              Column('asset_type', String),
              Column('beta', Numeric)
              )

fundamental = Table('fundamental', metadata,
                    Column('id', Integer, primary_key=True),
                    Column('ticker', Integer, ForeignKey=('asset.c.ticker')),
                    Column('amended', Boolean),
                    Column('assets', Numeric),
                    Column('current_assets', Numeric),
                    Column('current_liabilities', Numeric),
                    Column('cash', Numeric),
                    Column('dividend', Numeric),
                    Column('end_date', DateTime),
                    Column('eps', Numeric),
                    Column('eps_diluted', Numeric),
                    Column('equity', Numeric),
                    Column('net_income', Numeric),
                    Column('operating_income', Numeric),
                    Column('revenues', Numeric),
                    Column('investment_revenues', Numeric),
                    Column('fin_cash_flow', Numeric),
                    Column('inv_cash_flow', Numeric),
                    Column('ops_cash_flow', Numeric),
                    Column('year', String),
                    Column('period_focus', String),
                    Column('property_plant_equipment', Numeric),
                    Column('gross_profit', Numeric),
                    Column('tax_expense', Numeric),
                    Column('net_taxes_paid', Numeric),
                    Column('acts_pay_current', Numeric),
                    Column('acts_receive_current', Numeric),
                    Column('acts_receive_noncurrent', Numeric),
                    Column('acts_receive', Numeric),
                    Column('accrued_liabilities_current', Numeric),
                    Column('inventory_net', Numeric),
                    Column('interest_expense', Numeric),
                    Column('total_liabilities', Numeric),
                    Column('total_liabilities_equity', Numeric),
                    Column('shares_outstanding', Numeric),
                    Column('shares_outstanding_diluted', Numeric),
                    Column('depreciation_amortization', Numeric),
                    Column('cogs', Numeric),
                    Column('comprehensive_income_net_of_tax', Numeric),
                    Column('research_and_dev_expense', Numeric),
                    Column('common_stock_outstanding', Numeric),
                    Column('warranty_accrual', Numeric),
                    Column('warranty_accrual_payments', Numeric),
                    Column('ebit', Numeric),
                    Column('ebitda', Numeric),
                    Column('ticker', String),
                    )

portfolio = Table('portfolio', metadata,
                  Column('id ', Integer, primary_key=True),
                  Column('cash', Numeric),
                  Column('benchmark_ticker', String)
                  )

blotter = Table('blotter', metadata,
                Column('id', Integer, primary_key=True),
                Column('portfolio_id', Integer, ForeignKey=('portfolio.c.id'))
                )

order = Table('order', metadata,
              Column('id ', Integer, primary_key=True),
              Column('blotter_id', Integer, ForeignKey=('blotter.c.id'), primary_key=True),
              Column('asset_id', Integer, ForeignKey=('asset.c.id'), primary_key=True),
              Column('asset_type', String),
              Column('status', String),
              Column('created', Integer),
              Column('close_date', DateTime),
              Column('commission', Numeric),
              Column('stop_price', Numeric),
              Column('limit_price', Numeric),
              Column('stop_reached', Boolean),
              Column('limit_reached', Boolean),
              Column('qty', Integer),
              Column('filled', Integer),
              Column('action', String),
              Column('reason', String),
              Column('order_type', String)
              )

trade = Table('trade', metadata,
              Column('id', Integer, primary_key=True),
              Column('trade_date', Integer),
              Column('action', String),
              Column('strategy', String),
              Column('qty', Integer),
              Column('corresponding_trade_id', Integer, ForeignKey=('trade.c.id')),
              Column('net_trade_value', Numeric),
              Column('ticker', String),
              Column('order_id ', Integer, ForeignKey=('order.c.id')),
              Column('commission', Integer)
              )

owned_asset = Table('owned_asset', metadata,
                    Column('id', Integer, primary_key=True),
                    Column('asset_id', Integer, ForeignKey=('asset.c.id'), primary_key=True),
                    Column('asset_type', String),
                    Column('portfolio_id', Integer, ForeignKey=('portfolio.c.id'), primary_key=True),
                    Column('purchase_date ', Integer),
                    Column('average_share_price_paid', Numeric),
                    Column('shares_owned', Integer),
                    Column('total_position_value', Numeric)
                    )

universe_ohlcv = Table('universe_ohlcv', metadata,
                       Column('ticker', Integer, primary_key=True, ForeignKey=('asset.c.ticker')),
                       Column('asof_date', Integer, primary_key=True),
                       Column('open', Numeric),
                       Column('high', Numeric),
                       Column('low', Numeric),
                       Column('close', Numeric),
                       Column('adj_close', Numeric),
                       Column('volume', Numeric)
                       )

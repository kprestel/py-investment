import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import (
    ENUM,
    TEXT,
    TIMESTAMP,
    UUID,
)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement


class utcnow(FunctionElement):
    type = TIMESTAMP()


@compiles(utcnow, 'postgresql')
def pg_utcnow(element, compiler, **kwargs):
    return "TIMEZONE('utc', CURRENT_TIMESTAMP)"


from pytech.utils.enums import (
    TradeAction,
    OrderType,
    OrderSubType,
)

metadata = sa.MetaData()

trade_actions = ENUM(TradeAction, name='trade_actions', metadata=metadata)
order_types = ENUM(OrderType, name='order_types', metadata=metadata)
order_sub_types = ENUM(OrderSubType, name='order_sub_types', metadata=metadata)

tables = frozenset({
    'asset',
    'orders',
    'trade',
    'bar',
    'portfolio',
    'transaction',
    'ownership'
})

assets = sa.Table(
    'asset',
    metadata,
    sa.Column('ticker', sa.String, primary_key=True),
    sa.Column('exchange', sa.String),
    sa.Column('created', TIMESTAMP(timezone=True), server_default=utcnow()),
    sa.Column('last_updated', TIMESTAMP(timezone=True),
              server_onupdate=utcnow()),
)

orders = sa.Table(
    'order',
    metadata,
    sa.Column('id', UUID, primary_key=True, default=uuid.uuid4(), unique=True),
    sa.Column('ticker', None, sa.ForeignKey('asset.ticker'), primary_key=True),
    sa.Column('action', trade_actions, nullable=False),
    sa.Column('qty', sa.INTEGER, nullable=False),
    sa.Column('type', order_types, nullable=False),
    sa.Column('sub_type', order_sub_types),
    sa.Column('max_days_open', sa.INTEGER, default=1),
    sa.Column('close_date', TIMESTAMP(timezone=True)),
    sa.Column('filled', sa.INTEGER, default=0),
    sa.Column('stop_price', sa.Numeric(16, 2)),
    sa.Column('limit_price', sa.Numeric(16, 2)),
    sa.Column('created', TIMESTAMP(timezone=True), server_default=utcnow()),
    sa.Column('last_updated', TIMESTAMP(timezone=True),
              server_onupdate=utcnow()),
    sa.UniqueConstraint('id', 'ticker', name='uix_order_ticker')
)

trade = sa.Table(
    'trade',
    metadata,
    sa.Column('id', UUID, primary_key=True, default=uuid.uuid4(), unique=True),
    sa.Column('order_id', UUID, sa.ForeignKey('order.id'), primary_key=True),
    sa.Column('qty', sa.INTEGER, nullable=False),
    sa.Column('price_per_share', sa.INTEGER, nullable=False),
    sa.Column('commission', sa.Numeric(16, 2)),
    sa.Column('strategy', TEXT),
    sa.Column('created', TIMESTAMP(timezone=True), server_default=utcnow()),
    sa.Column('last_updated', TIMESTAMP(timezone=True),
              server_onupdate=utcnow()),
    sa.UniqueConstraint('id', 'order_id', name='uix_id_order_id')
)

bars = sa.Table(
    'bar',
    metadata,
    sa.Column('date', TIMESTAMP, nullable=False),
    sa.Column('open', sa.Numeric(16, 2), nullable=False),
    sa.Column('high', sa.Numeric(16, 2), nullable=False),
    sa.Column('low', sa.Numeric(16, 2), nullable=False),
    sa.Column('close', sa.Numeric(16, 2), nullable=False),
    sa.Column('volume', sa.INTEGER, nullable=False),
    sa.Column('adj_close', sa.Numeric(16, 2)),  # not all sources have this
    sa.Column('ticker', None, sa.ForeignKey('asset.ticker'), nullable=False),
    sa.UniqueConstraint('date', 'ticker', name='uix_date_ticker')
)

# after creating the bar table, turn it into a hyper table.
# event.listen(bars, 'after_create',
#              sa.DDL('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;\n'
#                     'SELECT create_hypertable("bar", "date");'))

portfolio = sa.Table(
    'portfolio',
    metadata,
    sa.Column('id', UUID, primary_key=True, default=uuid.uuid4().hex),
    sa.Column('cash', sa.Numeric(16, 2), nullable=False),
    sa.Column('initial_capital', sa.Numeric(16, 2), nullable=False),
    sa.Column('created', TIMESTAMP(timezone=True), server_default=utcnow()),
    sa.Column('last_updated', TIMESTAMP(timezone=True),
              server_onupdate=utcnow()),
)

transaction = sa.Table(
    'transaction',
    metadata,
    sa.Column('id', UUID, primary_key=True, default=uuid.uuid4()),
    sa.Column('portfolio_id', None, sa.ForeignKey('portfolio.id'),
              primary_key=True),
    sa.Column('trade_id', None, sa.ForeignKey('trade.id')),
    sa.Column('amount', sa.Numeric(16, 2), nullable=False),
    sa.Column('created', TIMESTAMP(timezone=True), server_default=utcnow()),
    sa.Column('last_updated', TIMESTAMP(timezone=True),
              server_onupdate=utcnow()),
)

ownership = sa.Table(
    'ownership',
    metadata,
    sa.Column('portfolio_id', None, sa.ForeignKey('portfolio.id'),
              primary_key=True),
    sa.Column('ticker', None, sa.ForeignKey('asset.ticker'), primary_key=True),
    sa.Column('shares_owned', sa.INTEGER, nullable=False),
    sa.Column('avg_price_per_share', sa.Numeric(16, 2), nullable=False),
    sa.Column('commission', sa.Numeric(16, 2)),
    sa.Column('created', TIMESTAMP(timezone=True), server_default=utcnow()),
    sa.Column('last_updated', TIMESTAMP(timezone=True),
              server_onupdate=utcnow()),
    sa.UniqueConstraint('portfolio_id', 'ticker', name='uix_ownership_pk')
)

asset_snapshot = sa.Table(
    'asset_snapshot',
    metadata,
    sa.Column('portfolio_id', None, sa.ForeignKey('portfolio.id'),
              primary_key=True),
    sa.Column('ticker', None, sa.ForeignKey('asset.ticker'), primary_key=True),
    sa.Column('date', TIMESTAMP(timezone=TIMESTAMP), primary_key=True),
    sa.Column('shares', sa.INTEGER),
    sa.Column('mv', sa.Numeric(16, 2)),
    sa.Column('close', sa.Numeric(16, 2))
)

portfolio_snapshot = sa.Table(
    'portfolio_snapshot',
    metadata,
    sa.Column('portfolio_id', None, sa.ForeignKey('portfolio.id'),
              primary_key=True),
    sa.Column('date', TIMESTAMP(timezone=TIMESTAMP), primary_key=True),
    sa.Column('cash', sa.Numeric(16, 2)),
    sa.Column('equity', sa.Numeric(16, 2)),
    sa.Column('commission', sa.Numeric(16, 2))
)

if __name__ == '__main__':
    engine = sa.create_engine('postgresql://pytech:pytech@localhost/pytech',
                              echo=True)
    metadata.create_all(engine, checkfirst=True)

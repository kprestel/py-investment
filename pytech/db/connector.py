from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from pytech.db.pytech_db_schema import (
    metadata,
    asset as asset_table,
    PYTECH_DB_TABLE_NAMES,
    owned_asset as owned_asset_table,
    trade as trade_table,
    portfolio as portfolio_table,
    order as order_table,
    universe_ohlcv
)


class DBConnector(object):
    """Handles creating the DB Connection and Creating the DB."""
    def __init__(self, engine, **kwargs):

        if engine is None:
            self.engine = create_engine('sqlite:///:memory:', **kwargs)
        elif isinstance(engine, Engine):
            self.engine = engine
        else:
            raise TypeError(
                'Engine must be a URI to a database or a SQLAlchemy Engine.')

        self.init_db()

    def init_db(self):
        metadata.create_all(self.engine, checkfirst=True)

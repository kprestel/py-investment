from sqlalchemy import create_engine
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

    def __init__(self, db_path):

        if db_path is None:
            self.engine = create_engine('sqlite:///:memory:')
        else:
            self.engine = create_engine(db_path)

    def init_db(self):
        metadata.create_all(self.engine)


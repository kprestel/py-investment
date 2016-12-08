from os.path import dirname, join

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
# from pytech.portfolio import Portfolio
import logging
PROJECT_DIR = dirname(__file__)
DATABASE_LOCATION = join(PROJECT_DIR, 'pytech.sqlite')
cs = 'sqlite+pysqlite:///{}'.format(DATABASE_LOCATION)
engine = create_engine(cs, connect_args={'check_same_thread':False}, poolclass=StaticPool)

# Session must be created before importing the other classes.
Session = sessionmaker(bind=engine)
from pytech.base import Base
from pytech.portfolio import Portfolio, Trade
from pytech.stock import Stock, Fundamental, HasStock, PortfolioAsset, OwnedStock
# Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
logging.basicConfig(level=logging.DEBUG)

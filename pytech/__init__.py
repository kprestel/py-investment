from os.path import dirname, join, pardir
import os

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
# from pytech.portfolio import Portfolio
import logging
PROJECT_DIR = dirname(__file__)
RESOURCE_DIR = join(pardir, 'resources')
DATA_DIR = join(RESOURCE_DIR, 'data')

try:
    os.makedirs(RESOURCE_DIR)
except OSError:
    pass

try:
    os.makedirs(DATA_DIR)
except OSError:
    pass

DATABASE_LOCATION = join(RESOURCE_DIR, 'pytech.sqlite')
cs = 'sqlite+pysqlite:///{}'.format(DATABASE_LOCATION)
engine = create_engine(cs, connect_args={'check_same_thread':False}, poolclass=StaticPool)

# Session must be created before importing the other classes.
Session = sessionmaker(bind=engine)
from pytech.base import Base
from pytech.portfolio import Portfolio
from pytech.order import Trade
from pytech.asset import Stock, Fundamental, HasStock, PortfolioAsset, OwnedAsset
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
logging.basicConfig(level=logging.DEBUG)

from os.path import dirname, join

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from pytech.portfolio import Portfolio
import logging
from pytech.stock import Base

PROJECT_DIR = dirname(__file__)

DATABASE_LOCATION = join(PROJECT_DIR, 'pytech.db')

logging.basicConfig(level=logging.DEBUG)

cs = 'sqlite+pysqlite:///{}'.format(DATABASE_LOCATION)
engine = create_engine(cs, connect_args={'check_same_thread':False}, poolclass=StaticPool)
Session = sessionmaker(bind=engine)
# Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

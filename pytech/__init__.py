import logging
import os
from os.path import dirname, join, pardir

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_DIR = dirname(__file__)
RESOURCE_DIR = join(pardir, 'resources')
DATA_DIR = join(RESOURCE_DIR, 'data')
TEST_DATA_DIR = join(pardir, 'tests', 'sample_data', 'csv')

try:
    os.makedirs(RESOURCE_DIR)
except OSError:
    pass

try:
    os.makedirs(DATA_DIR)
except OSError:
    pass

# DATABASE_LOCATION = join(RESOURCE_DIR, 'pytech.sqlite')
# cs = 'sqlite+pysqlite:///{}'.format(DATABASE_LOCATION)
# engine = create_engine(cs, connect_args={'check_same_thread':False}, poolclass=StaticPool)

# Session must be created before importing the other classes.
# Session = sessionmaker(bind=engine)
logging.basicConfig(level=logging.INFO)

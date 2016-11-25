from os.path import dirname, join

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_DIR = dirname(__file__)

DATABASE_LOCATION = join(PROJECT_DIR, 'pytech.db')

cs = 'sqlite+pysqlite:///{}'.format(DATABASE_LOCATION)
engine = create_engine(cs)
Session = sessionmaker(bind=engine)
import sqlalchemy as sa
from .connection import getconn, write, reader
from .schema import metadata
from .reader import BarReader
from . import handler

metadata.create_all(sa.create_engine('postgresql+psycopg2://',
                                     creator=getconn), checkfirst=True)

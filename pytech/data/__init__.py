import sqlalchemy as sa
from .schema import metadata
from .connection import getconn
from .handler import BarReader, DataHandler, Bars

# _engine = sa.create_engine('postgresql+psycopg2://', creator=getconn,
#                            echo=True)
metadata.create_all(sa.create_engine('postgresql+psycopg2://',
                                     creator=getconn), checkfirst=True)

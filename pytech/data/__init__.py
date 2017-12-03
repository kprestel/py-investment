import sqlalchemy as sa

from ._holders import ReaderResult
from .connection import (
    getconn,
    reader,
    write,
)
from .reader import BarReader
from .schema import metadata

metadata.create_all(sa.create_engine('postgresql+psycopg2://',
                                     creator=getconn), checkfirst=True)

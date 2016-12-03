import re

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy.ext.declarative import as_declarative
from sqlalchemy.ext.declarative import declared_attr


@as_declarative(constructor=None)
class Base(object):
    @declared_attr
    def __tablename__(cls):
        name = cls.__name__
        return (
            name[0].lower() +
            re.sub(r'([A-Z])',
                   lambda m: '_' + m.group(0).lower(), name[1:])
        )

    id = Column(Integer, primary_key=True)


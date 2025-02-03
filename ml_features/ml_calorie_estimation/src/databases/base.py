from sqlalchemy import Column, Integer
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class BaseTable(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True)
    # other common columns...

    @classmethod
    def from_dict(cls, data: dict) -> 'BaseTable':
        # common logic for creating an instance from a dictionary
        pass
from typing import List, Type
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ml_features.ml_calorie_estimation.src.databases.models.raw_data import Base
from ml_features.ml_calorie_estimation.src.databases.config import DatabaseConfig

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_engine(config.connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
    def init_db(self):
        """Initialize database tables"""
        Base.metadata.create_all(self.engine)
        
    def store_records(self, records: List[dict], model_class: Type):
        """Store multiple records in the database
        
        Args:
            records: List of dictionaries containing record data
            model_class: SQLAlchemy model class to use for storing records
        """
        session = self.Session()
        try:
            for i, record in enumerate(records):
                try:
                    db_record = model_class.from_dict(record)
                    session.add(db_record)
                except Exception as e:
                    logger.error(f"Failed to store record at index {i}: {e}")
                    continue
            session.commit()
        except Exception as e:
            logger.error(f"Database transaction failed: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def delete_all_records(self, model_class: Type):
        """Delete all records from specified table
        
        Args:
            model_class: SQLAlchemy model class representing the table to clear
        """
        session = self.Session()
        try:
            session.query(model_class).delete()
            session.commit()
        except Exception as e:
            logger.error(f"Failed to delete records from {model_class.__name__}: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
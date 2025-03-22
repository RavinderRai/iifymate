from typing import List, Type, TypeVar
import logging
import time
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from ml_features.ml_calorie_estimation.src.databases.config import DatabaseConfig
from ml_features.ml_calorie_estimation.src.databases.base import BaseTable

T = TypeVar('T', bound= BaseTable)

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_engine(
            config.connection_string
        )
        self.Session = sessionmaker(bind=self.engine)
        
    def init_db(self):
        """Initialize database tables"""
        max_retries = 1
        for attempt in range(max_retries):
            try:
                BaseTable.metadata.create_all(self.engine)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to initialize database after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Database initialization attempt {attempt + 1} failed: {e}")
                time.sleep(5)
        
    def create_table(self, model_class: Type[T]) -> None:
        """Create table if it doesn't exist
        
        Args:
            model_class: SQLAlchemy model class extending BaseTable
        """
        inspector = inspect(self.engine)
        if not inspector.has_table(model_class.__tablename__):
            model_class.__table__.create(self.engine)
            logger.info(f"Created table {model_class.__tablename__}")
        
    def store_records(self, records: List[dict], model_class: Type[T]) -> int:
        """Store multiple records in the database
        
        Args:
            records: List of dictionaries containing record data
            model_class: SQLAlchemy model class to use for storing records
            
        Returns:
            int: Number of records successfully stored
        """
        session = self.Session()
        successful_records = 0
        try:
            for i, record in enumerate(records):
                try:
                    db_record = model_class.from_dict(record)
                    session.add(db_record)
                    successful_records += 1
                except Exception as e:
                    logger.error(f"Failed to store record at index {i}: {e}")
                    continue
            session.commit()
            return successful_records
        except Exception as e:
            logger.error(f"Database transaction failed: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def delete_all_records(self, model_class: Type[T]) -> None:
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
    
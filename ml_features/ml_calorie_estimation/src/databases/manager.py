from typing import List
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ml_features.ml_calorie_estimation.src.databases.models.raw_data import Base, RawRecipe
from ml_features.ml_calorie_estimation.src.databases.config import DatabaseConfig

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.engine = create_engine(config.connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
    def init_db(self):
        """Initialize database tables"""
        Base.metadata.create_all(self.engine)
        
    def store_recipes(self, recipes: List[dict]):
        """Store multiple recipes in the database"""
        session = self.Session()
        try:
            for i, recipe in enumerate(recipes):
                try:
                    raw_recipe = RawRecipe.from_dict(recipe)
                    session.add(raw_recipe)
                except Exception as e:
                    logger.error(f"Failed to store recipe at index {i}: {e}")
                    continue
            session.commit()
        except Exception as e:
            logger.error(f"Database transaction failed: {e}")
            session.rollback()
            raise
        finally:
            session.close()


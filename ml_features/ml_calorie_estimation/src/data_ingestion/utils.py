import os
from dotenv import load_dotenv
from typing import Literal
import yaml
import logging
from pathlib import Path
from ml_features.ml_calorie_estimation.src.databases.config import DatabaseConfig
from ml_features.ml_calorie_estimation.src.data_ingestion.config import APIConfig, RecipeParameters, CollectorConfig, Config
from ml_features.ml_calorie_estimation.src.data_ingestion.constants import DIET_LABELS, HEALTH_LABELS, MEAL_TYPES, DISH_TYPES, CUISINE_TYPES

load_dotenv()

logger = logging.getLogger(__name__)
    
def load_config(env: Literal["local", "production"] = "local") -> Config:
   """Load configuration based on environment
   
   Args:
       env: Environment to load config for ('local' or 'production')
       
   Returns:
       Config object with environment-specific settings
       
   Raises:
       FileNotFoundError: If config file doesn't exist
       KeyError: If required environment variables are missing
   """
   config_path = Path(f"ml_features/ml_calorie_estimation/configs/{env}.yaml")
   
   logger.info(f"Loading {env} configuration from {config_path}")
   
   try:
       with open(config_path, 'r') as f:
           config_dict = yaml.safe_load(f)
   except FileNotFoundError:
       logger.error(f"Configuration file not found: {config_path}")
       raise
   
   # Load API credentials
   api_id = os.getenv("EDAMAM_API_ID")
   api_key = os.getenv("EDAMAM_API_KEY")
   
   if not api_id or not api_key:
       logger.error("Missing required API credentials in environment variables")
       raise KeyError("EDAMAM_API_ID and EDAMAM_API_KEY must be set")
       
   config_dict['api']['app_id'] = api_id
   config_dict['api']['app_key'] = api_key
   
   if env == "production":
       logger.info("Loading production database credentials")
       
       # Verify required environment variables exist
       required_vars = ["RDS_USERNAME", "RDS_PASSWORD", "RDS_HOST"]
       missing_vars = [var for var in required_vars if not os.getenv(var)]
       
       if missing_vars:
           logger.error(f"Missing required environment variables: {missing_vars}")
           raise KeyError(f"Missing required RDS credentials: {missing_vars}")
           
       # Update database config with RDS credentials
       config_dict['database'].update({
           "username": os.getenv("RDS_USERNAME"),
           "password": os.getenv("RDS_PASSWORD"),
           "host": os.getenv("RDS_HOST"),
       })
       
       logger.debug("Successfully loaded RDS configuration", extra={
           "host": os.getenv("RDS_HOST"),
           "username": os.getenv("RDS_USERNAME"),
           "database": config_dict['database'].get('database')
       })
   else:
       logger.info("Loading local database credentials")
       local_db_password = os.getenv("POSTGRESQL_IIFYMATE_PASSWORD")
       
       if not local_db_password:
           logger.error("Missing local database password")
           raise KeyError("POSTGRESQL_IIFYMATE_PASSWORD must be set")
           
       config_dict['database']['password'] = local_db_password
   
   logger.info(f"Successfully loaded {env} configuration")
   return Config(**config_dict)


def create_api_config(api_config: APIConfig) -> APIConfig:
    """Create API configuration from APIConfig object"""
    api_config.app_id = os.getenv("EDAMAM_API_ID")
    api_config.app_key = os.getenv("EDAMAM_API_KEY")
    return api_config

def create_collector_config(collector_config: CollectorConfig) -> CollectorConfig:
    """Create collector configuration from CollectorConfig object with additional default values"""
    # currently no additional default values
    return collector_config

def create_recipe_parameters() -> RecipeParameters:
    """Create recipe parameters with default values"""
    return RecipeParameters(
        diet_labels=DIET_LABELS,
        health_labels=HEALTH_LABELS,
        meal_types=MEAL_TYPES,
        dish_types=DISH_TYPES,
        cuisine_types=CUISINE_TYPES
    )
    
def create_db_config(db_config: DatabaseConfig, env:str = "local") -> DatabaseConfig:
    """Create database configuration from environment variables"""
    if env == "production":
        return DatabaseConfig(
            username=os.getenv("RDS_USERNAME"),
            password=os.getenv("RDS_PASSWORD"),
            host=os.getenv("RDS_HOST"),
            database=db_config.database,
            port=int(os.getenv("RDS_PORT", "5432")),
            ssl_mode="require", # Enable SSL for RDS
            env="production"
        )
    else:
        return DatabaseConfig(
            username=db_config.username,
            password=os.getenv("POSTGRESQL_IIFYMATE_PASSWORD"),
            host=db_config.host,
            database=db_config.database,
            port=5432,
            env="local"
        )

import os
from dotenv import load_dotenv
from typing import Literal
import yaml
from ml_features.ml_calorie_estimation.src.databases.config import DatabaseConfig
from ml_features.ml_calorie_estimation.src.data_ingestion.config import APIConfig, RecipeParameters, CollectorConfig, Config
from ml_features.ml_calorie_estimation.src.data_ingestion.constants import DIET_LABELS, HEALTH_LABELS, MEAL_TYPES, DISH_TYPES, CUISINE_TYPES

load_dotenv()
    
def load_config(env: Literal["local", "production"] = "local") -> Config:
    """Load configuration based on environment"""
    config_path = f"ml_features/ml_calorie_estimation/configs/{env}.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    config_dict['api']['app_id'] = os.getenv("EDAMAM_API_ID")
    config_dict['api']['app_key'] = os.getenv("EDAMAM_API_KEY")
    config_dict['database']['password'] = os.getenv("POSTGRESQL_IIFYMATE_PASSWORD")
    
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
    
def create_db_config(db_config: DatabaseConfig) -> DatabaseConfig:
    """Create database configuration from environment variables"""
    db_config.password = os.getenv("POSTGRESQL_IIFYMATE_PASSWORD")
    return db_config
import os
from dotenv import load_dotenv
from typing import Any
import yaml
from ml_features.ml_calorie_estimation.src.databases.config import DatabaseConfig
from ml_features.ml_calorie_estimation.src.data_ingestion.models import APIConfig, RecipeParameters, CollectorConfig
from ml_features.ml_calorie_estimation.src.data_ingestion.constants import DIET_LABELS, HEALTH_LABELS, MEAL_TYPES, DISH_TYPES, CUISINE_TYPES

load_dotenv()

def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_api_config() -> APIConfig:
    """Create API configuration from dictionary"""
    return APIConfig(
        app_id=os.getenv("EDAMAM_API_ID"),
        app_key=os.getenv("EDAMAM_API_KEY"),
    )

def create_collector_config() -> CollectorConfig:
    """Create collector configuration with default values"""
    return CollectorConfig()

def create_recipe_parameters() -> RecipeParameters:
    """Create recipe parameters with default values"""
    return RecipeParameters(
        diet_labels=DIET_LABELS,
        health_labels=HEALTH_LABELS,
        meal_types=MEAL_TYPES,
        dish_types=DISH_TYPES,
        cuisine_types=CUISINE_TYPES
    )
    
def create_db_config() -> DatabaseConfig:
    """Create database configuration from environment variables"""
    return DatabaseConfig(
        username="iifymate",
        password=os.getenv("POSTGRESQL_IIFYMATE_PASSWORD"),
        host="localhost",
        database="recipe_data"
    )
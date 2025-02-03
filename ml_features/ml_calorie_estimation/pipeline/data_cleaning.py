import pandas as pd
import logging
from ml_features.ml_calorie_estimation.src.data_ingestion.utils import (
    create_db_config,
    load_config
)
from ml_features.ml_calorie_estimation.src.data_cleaning import recipe_cleaner
from ml_features.ml_calorie_estimation.src.databases.manager import DatabaseManager
from ml_features.ml_calorie_estimation.src.databases.models.raw_data import RawRecipe
from ml_features.ml_calorie_estimation.src.databases.models.clean_data import CleanRecipe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to the recipe dataframe."""
    
    # Make a copy of the original ids
    original_ids = df['id'].copy()
    
    logger.info("Starting data cleaning process.")
    df = recipe_cleaner.remove_faulty_nutrients(df)
    logger.info("Removed faulty nutrients.")
    df = recipe_cleaner.drop_unused_columns(df)
    logger.info("Dropped unused columns.")
    df = recipe_cleaner.rename_columns(df)
    logger.info("Renamed columns.")
    df = recipe_cleaner.clean_ingredients(df)
    logger.info("Cleaned ingredients.")
    df = recipe_cleaner.convert_numeric_columns(df)
    logger.info("Converted numeric columns.")
    df = recipe_cleaner.clean_nutrients(df)
    logger.info("Cleaned nutrients.")
    
    # Add back the original ID as raw_recipe_id
    df['raw_recipe_id'] = original_ids
    
    logger.info("Data cleaning process completed.")
    return df

def clean_and_store_recipes(env: str = "local", delete_all_recipes: bool = False):
    logger.info(f"Loading configuration for environment: {env}")
    config = load_config(env)
    
    # Initialize database and load raw data
    logger.info("Creating database configuration.")
    db_config = create_db_config(config.database)
    db_manager = DatabaseManager(db_config)
    # Create the table if it doesn't exist
    db_manager.create_table(CleanRecipe)
    
    logger.info("Loading raw recipe data from database.")
    session = db_manager.Session()
    query = session.query(RawRecipe).statement
    df = pd.read_sql(query, session.bind)
    
    logger.info("Cleaning raw recipe data.")
    df = clean_data(df)
    
    # Ensure this is a list of dictionaries
    clean_recipes = df.to_dict(orient='records')
    
    if delete_all_recipes:
        logger.info("Deleting all existing clean recipes from database.")
        db_manager.delete_all_records(CleanRecipe)
        
    logger.info("Storing cleaned recipes into database.")
    num_stored = db_manager.store_records(clean_recipes, CleanRecipe)
    
    return num_stored

if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.pipeline.data_cleaning
    
    logger.info("Cleaning raw recipe data and storing in another table...")
    num_stored = clean_and_store_recipes(env="local", delete_all_recipes=True)
    logger.info(f"Successfully cleaned and stored {num_stored} records")
    
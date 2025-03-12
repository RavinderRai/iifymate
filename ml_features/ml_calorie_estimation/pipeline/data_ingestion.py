import asyncio
import logging
from ml_features.ml_calorie_estimation.src.utils import (
    create_api_config,
    create_collector_config,
    create_recipe_parameters,
    create_db_config,
    load_config
)
from ml_features.ml_calorie_estimation.src.data_ingestion.clients import EdamamClient
from ml_features.ml_calorie_estimation.src.data_ingestion.collectors import RecipeDataCollector
from ml_features.ml_calorie_estimation.src.databases.manager import DatabaseManager
from ml_features.ml_calorie_estimation.src.databases.models.raw_data import RawRecipe
from ml_features.ml_calorie_estimation.src.databases.models.clean_data import CleanRecipe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def collect_and_store_recipes(env: str = "local", delete_all_recipes: bool = False):
    """Main function to collect and store recipes"""
    logger.info(f"Loading configuration for environment: {env}")
    config = load_config(env)
    
    # Initialize configurations
    api_config = create_api_config(config.api)
    collector_config = create_collector_config(config.collection)
    recipe_params = create_recipe_parameters()
    db_config = create_db_config(config.database, env=env)
    
    semaphore = asyncio.Semaphore(collector_config.requests_per_minute)
    
    # Initialize components
    client = EdamamClient(api_config)
    collector = RecipeDataCollector(
        params=recipe_params,
        client=client,
        min_recipes_per_category=collector_config.min_recipes_per_category,
        max_retries=collector_config.max_retries,
        rate_limit=collector_config.requests_per_minute,
        semaphore=semaphore
    )
    
    # Collect recipes
    # Note it is important to do this first before initializing the database, 
    # otherwise the connection will timeout and the data will not be stored. 
    # Also, make sure to modify collection later so that the data is stored as it's collected instead of all at once
    recipes = await collector.collect_recipes(
        target_recipes=collector_config.target_recipes
    )
    
    db_manager = DatabaseManager(db_config)
    db_manager.init_db()
    # Check if table already exists and create if not
    db_manager.create_table(RawRecipe)
    
    if delete_all_recipes:
        logger.info("Deleting all recipes from database...")
        try:
            # try to delete all clean recipes, just in case this table hasn't been created yet
            db_manager.delete_all_records(CleanRecipe) # Note: you must delete this first to prevent IntegrityError error due to relationship
        except Exception as e:
            logger.error(f"Failed to delete clean recipes from database: {e}")
        
        db_manager.delete_all_records(RawRecipe)
        logger.info("Successfully deleted all recipes from database")
    
    logger.info("Storing recipes in database...")
    num_stored = db_manager.store_records(recipes, RawRecipe)
    
    logger.info(f"Successfully stored {num_stored} recipes in database")

    return recipes, collector.parameter_stats

if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.pipeline.data_ingestion
    import os
    environment = os.getenv("ENV", "local")
    recipes, stats = asyncio.run(collect_and_store_recipes(env=environment, delete_all_recipes=True))
    
    # Display statistics only when run as main script
    logger.info("\n=== Collection Statistics ===")
    logger.info(f"Total Recipes Collected: {len(recipes)}")
    logger.info("\nParameter Success Rates:")
    for param_type, param_stats in stats.items():
        success_rate = param_stats.success / param_stats.total if param_stats.total > 0 else 0
        logger.info(f"{param_type:15}: {success_rate:.2%} ({param_stats.success}/{param_stats.total})")
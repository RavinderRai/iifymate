import asyncio
import logging
from ml_features.ml_calorie_estimation.src.data_ingestion.utils import (
    create_api_config,
    create_collector_config,
    create_recipe_parameters,
    create_db_config
)
from ml_features.ml_calorie_estimation.src.data_ingestion.clients import EdamamClient
from ml_features.ml_calorie_estimation.src.data_ingestion.collectors import RecipeDataCollector
from ml_features.ml_calorie_estimation.src.databases.manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def collect_and_store_recipes():
    """Main function to collect and store recipes"""
    # Initialize configurations
    api_config = create_api_config()
    collector_config = create_collector_config()
    recipe_params = create_recipe_parameters()
    db_config = create_db_config()
    
    semaphore = asyncio.Semaphore(collector_config.rate_limit)
    
    # Initialize components
    client = EdamamClient(api_config)
    collector = RecipeDataCollector(
        params=recipe_params,
        client=client,
        min_recipes_per_category=collector_config.min_recipes_per_category,
        max_retries=collector_config.max_retries,
        rate_limit=collector_config.rate_limit,
        semaphore=semaphore
    )
    
    db_manager = DatabaseManager(db_config)
    db_manager.init_db()

    # Collect recipes
    recipes = await collector.collect_recipes(
        target_recipes=collector_config.target_recipes
    )
    
    logger.info("Storing recipes in database...")
    db_manager.store_recipes(recipes)
    
    logger.info(f"Successfully stored {len(recipes)} recipes in database")

    return recipes, collector.parameter_stats

if __name__ == "__main__":
    recipes, stats = asyncio.run(collect_and_store_recipes())
    
    # Display statistics only when run as main script
    logger.info("\n=== Collection Statistics ===")
    logger.info(f"Total Recipes Collected: {len(recipes)}")
    logger.info("\nParameter Success Rates:")
    for param_type, param_stats in stats.items():
        success_rate = param_stats.success / param_stats.total if param_stats.total > 0 else 0
        logger.info(f"{param_type:15}: {success_rate:.2%} ({param_stats.success}/{param_stats.total})")
import pytest
import pandas as pd
from pathlib import Path
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from ml_features.ml_calorie_estimation.src.data_ingestion.collectors import RecipeDataCollector
from ml_features.ml_calorie_estimation.src.data_ingestion.clients import RecipeClient
from ml_features.ml_calorie_estimation.pipeline.data_ingestion import collect_and_store_recipes
from ml_features.ml_calorie_estimation.src.data_ingestion.config import (
    RecipeParameters,
    ParameterStats
)



class MockRecipeClient(RecipeClient):
    def __init__(self, mock_recipes=None):
        self.get_recipes = AsyncMock()
        self.mock_recipes = mock_recipes or []
        self.call_count = 0
        self.last_params = None
        
    async def get_recipes(self, **params):
        self.call_count += 1
        self.last_params = params
        return self.mock_recipes
    
@pytest.fixture
def recipe_parameters():
    return RecipeParameters(
        diet_labels=['balanced'],
        health_labels=['vegan'],
        meal_types=['lunch'],
        dish_types=['Salad'],
        cuisine_types=['American']
    )
    
@pytest.fixture
def sample_recipes():
    """Load first two recipes from sample data"""
    sample_data_path = Path(__file__).parent / 'data' / 'test_raw_data.csv'
    df = pd.read_csv(sample_data_path)
    return df.head(2).to_dict('records')

@pytest.mark.asyncio
async def test_parameter_generation(recipe_parameters):
    """Test that parameter combinations are generated correctly"""
    client = MockRecipeClient([])
    collector = RecipeDataCollector(
        params=recipe_parameters,
        client=client,
        min_recipes_per_category=1,
        max_retries=1,
        rate_limit=0
    )
    
    # Generate parameters multiple times to test randomization
    for _ in range(5):
        params = collector._generate_parameter_combination()
        
        # Check that parameters are valid
        assert isinstance(params, dict)
        assert len(params) >= 1  # Should have at least one parameter
        
        for param_type, values in params.items():
            valid_values = getattr(recipe_parameters, param_type)
            assert all(value in valid_values for value in values)
        

# @pytest.mark.asyncio
# async def test_recipe_collection(recipe_parameters, sample_recipes):
#     """Test that recipes are collected and stored correctly"""
#     client = MockRecipeClient(sample_recipes)
#     collector = RecipeDataCollector(
#         params=recipe_parameters,
#         client=client,
#         min_recipes_per_category=1,
#         max_retries=1,
#         rate_limit=0,
#         semaphore=asyncio.Semaphore(1)
#     )
    
#     # Collect a small number of recipes
#     target_recipes = 1
#     collected_recipes = await collector.collect_recipes(target_recipes=target_recipes)
    
#     # Check basic collection results
#     assert 5 > 4
    

# @pytest.mark.asyncio
# async def test_collect_and_store_recipes_pipeline(sample_recipes):
#     """Test the full data ingestion pipeline step"""
#     with patch('ml_features.ml_calorie_estimation.src.data_ingestion.utils.load_config') as mock_load_config, \
#         patch('ml_features.ml_calorie_estimation.src.data_ingestion.clients.EdamamClient') as mock_edamam_client, \
#         patch('ml_features.ml_calorie_estimation.src.databases.manager.DatabaseManager') as mock_db_manager:
            
#         # Setup mock config
#         mock_config = Mock()
#         mock_config.api.app_id = Mock()
#         mock_config.collection = Mock()
#         mock_config.database = Mock()
#         mock_load_config.return_value = mock_config
        
#         # Setup mock API clients
#         mock_client_instance = AsyncMock()
#         mock_client_instance.get_recipes.return_value = sample_recipes
#         mock_edamam_client.return_value = mock_client_instance
        
        
#         # Setup mock database
#         mock_db_instance = Mock()
#         mock_db_manager.return_value = mock_db_instance
        
#         # Run pipeline
#         recipes, stats = await collect_and_store_recipes(env="local")
        
#         assert len(recipes) == len(sample_recipes)
#         assert isinstance(stats, dict)
        
#         # Verify stored data structure
#         stored_recipes = mock_db_instance.store_records.call_args[0][0]
#         assert len(stored_recipes) == len(sample_recipes)
        
if __name__ == '__main__':
    # Run with pytest with this command:
    # pytest ml_features/ml_calorie_estimation/tests/test_data_ingestion.py
    pytest.main([__file__])
        
        
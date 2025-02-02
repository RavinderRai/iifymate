import time
import asyncio
import random
from ml_features.ml_calorie_estimation.src.data_ingestion.config import RecipeParameters, ParameterStats
from ml_features.ml_calorie_estimation.src.data_ingestion.clients import RecipeClient


class RecipeDataCollector:
    def __init__(self,
                 params: RecipeParameters,
                 client: RecipeClient,
                 min_recipes_per_category: int = 100,
                 max_retries: int = 3,
                 rate_limit: float = 1.0,
                 semaphore: asyncio.Semaphore = None):
        self.params = params
        self.client = client
        self.min_recipes = min_recipes_per_category
        self.max_retries = max_retries
        self.semaphore = semaphore or asyncio.Semaphore(10)
        self.rate_limit = rate_limit
        self.collected_recipes: dict[str, dict] = {}
        self.parameter_stats: dict[str, ParameterStats] = {}
        
    def _update_stats(self, params: dict[str, list[str]], success: bool):
        """Update success statistics for parameter combinations"""
        for param_type in params:
            if param_type not in self.parameter_stats:
                self.parameter_stats[param_type] = ParameterStats()
                
            stats = self.parameter_stats[param_type]
            stats.total += 1
            if success:
                stats.success += 1
                
    def _store_recipe(self, recipe: dict):
        """Store recipe with additional metadata."""
        recipe_id = recipe['uri']
        self.collected_recipes[recipe_id] = {
            **recipe,
            'collection_timestamp': time.time(),
            'parameter_stats': self.parameter_stats.copy()
        }
    
    def _get_param_success_rate(self, param_type: str) -> float:
        """Calculate success rate for a parameter type"""
        if param_type not in self.parameter_stats:
            return 0.5
        
        stats = self.parameter_stats[param_type]
        return stats.success / stats.total if stats.total > 0 else 0.5
    
    def _generate_parameter_combination(self) -> dict[str, list[str]]:
            """
            Generate a smart parameter combination based on historical success rates.
            
            Returns:
                Dictionary of parameters to use in the next query
            """
            params = {}
            
            # Available parameter types
            param_types = ['diet_labels', 'health_labels', 'meal_types', 
                        'dish_types', 'cuisine_types']
            
            # Start with one random parameter type
            primary_param = random.choice(param_types)
            param_values = getattr(self.params, primary_param)
            params[primary_param] = [random.choice(param_values)]
            
            # Maybe add more parameters based on their success rates
            for param_type in param_types:
                if param_type != primary_param:
                    success_rate = self._get_param_success_rate(param_type)
                    if random.random() < success_rate:
                        param_values = getattr(self.params, param_type)
                        params[param_type] = [random.choice(param_values)]
                        
            return params
        
    async def collect_recipes(self, target_recipes: int = 1000) -> list[dict]:
        """
        Collect recipes using adaptive parameter sampling.
        
        Args:
            target_recipes: Number of unique recipes to collect
            
        Returns:
            List of collected recipes
        """
        unique_recipes = set()
        retries = 0
        
        while len(unique_recipes) < target_recipes and retries < self.max_retries:
            try:
                async with self.semaphore:
                    # Generate parameter combination
                    params = self._generate_parameter_combination()
                    print(f"Trying parameters: {params}")
                    
                    # Rate limiting
                    time.sleep(self.rate_limit)
                    
                    # Get recipes
                    recipes = await self.client.get_recipes(**params)
                    
                    if recipes:
                        # Update stats for successful combination
                        self._update_stats(params, success=True)
                        
                        # Store new unique recipes
                        for recipe in recipes:
                            recipe_id = recipe['uri']
                            if recipe_id not in unique_recipes:
                                unique_recipes.add(recipe_id)
                                self._store_recipe(recipe)
                    else:
                        # Update stats for unsuccessful combination
                        self._update_stats(params, success=False)
                        
                    print(f"Collected {len(unique_recipes)} unique recipes")
                
            except Exception as e:
                print(f"Error collecting recipes: {str(e)}")
                retries += 1
                
        return list(self.collected_recipes.values())
    
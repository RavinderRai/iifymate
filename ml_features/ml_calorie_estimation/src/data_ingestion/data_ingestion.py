import os
import requests
import pandas as pd
from pydantic import BaseModel, Field
import random
import time
import asyncio
import aiohttp
from typing import Any
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

edamam_api_id = os.environ["EDAMAM_API_ID"]
edamam_api_key = os.environ["EDAMAM_API_KEY"]

diet_labels = [
    "balanced",      # Protein/Fat/Carb values in 15/35/50 ratio
    "high-fiber",    # More than 5g fiber per serving
    "high-protein",  # More than 50% of total calories from proteins
    "low-carb",      # Less than 20% of total calories from carbs
    "low-fat",       # Less than 15% of total calories from fat
    "low-sodium"     # Less than 140mg Na per serving
]

health_labels = [
    "alcohol-cocktail",    # Describes an alcoholic cocktail
    "alcohol-free",        # No alcohol used or contained
    "celery-free",        # Does not contain celery or derivatives
    "crustacean-free",    # Does not contain crustaceans
    "dairy-free",         # No dairy; no lactose
    "DASH",               # Dietary Approaches to Stop Hypertension diet
    "egg-free",           # No eggs or products containing eggs
    "fish-free",          # No fish or fish derivatives
    "fodmap-free",        # Does not contain FODMAP foods
    "gluten-free",        # No ingredients containing gluten
    "immuno-supportive",  # Science-based immune system strengthening
    "keto-friendly",      # Maximum 7 grams of net carbs per serving
    "kidney-friendly",    # Restricted phosphorus, potassium, and sodium
    "kosher",             # Contains only kosher-allowed ingredients
    "low-potassium",      # Less than 150mg per serving
    "low-sugar",          # No simple sugars
    "lupine-free",        # Does not contain lupine or derivatives
    "Mediterranean",      # Mediterranean diet
    "mollusk-free",       # No mollusks
    "mustard-free",       # Does not contain mustard or derivatives
    "No-oil-added",       # No oil added except in basic ingredients
    "paleo",              # Excludes agricultural products
    "peanut-free",        # No peanuts or products containing peanuts
    "pecatarian",         # No meat, can contain dairy and fish
    "pork-free",          # Does not contain pork or derivatives
    "red-meat-free",      # No red meat or products containing red meat
    "sesame-free",        # Does not contain sesame seed or derivatives
    "shellfish-free",     # No shellfish or shellfish derivatives
    "soy-free",           # No soy or products containing soy
    "sugar-conscious",    # Less than 4g of sugar per serving
    "sulfite-free",       # No Sulfites
    "tree-nut-free",      # No tree nuts or products containing tree nuts
    "vegan",              # No animal products
    "vegetarian",         # No meat, poultry, or fish
    "wheat-free"          # No wheat, can have gluten though
]

meal_types = [
    "breakfast",
    "brunch",
    "lunch", # lunch/dinner are the same according to the API, so we only need one of these labels
    "snack",
    "teatime"
]

dish_types = [
    "alcohol cocktail",
    "biscuits and cookies",
    "bread",
    "cereals",
    "condiments and sauces",
    "desserts",
    "drinks",
    "egg",
    "ice cream and custard",
    "main course",
    "pancake",
    "pasta",
    "pastry",
    "pies and tarts",
    "pizza",
    "preps",
    "preserve",
    "salad",
    "sandwiches",
    "seafood",
    "side dish",
    "soup",
    "special occasions",
    "starter",
    "sweets"
]

cuisine_types = [
    "american",
    "asian",
    "british",
    "caribbean",
    "central europe",
    "chinese",
    "eastern europe",
    "french",
    "greek",
    "indian",
    "italian",
    "japanese",
    "korean",
    "kosher",
    "mediterranean",
    "mexican",
    "middle eastern",
    "nordic",
    "south american",
    "south east asian",
    "world" # International cuisine/Other
]


class RecipeParameters(BaseModel):
    """Stores all valid parameter values for recipe searches"""
    diet_labels: list[str] = Field(
        description="List of valid diet labels (e.g. 'balanced', 'high-protein', 'low-fat')",
    )
    health_labels: list[str] = Field(
        description="List of valid health labels (e.g. 'egg-free', 'gluten-free', 'peanut-free')",
    )
    meal_types: list[str] = Field(
        description="List of valid meal types (e.g. 'breakfast', 'lunch', 'dinner')",
    )
    dish_types: list[str] = Field(
        description="List of valid dish types (e.g. 'salad', 'main course', 'dessert')",
    )
    cuisine_types: list[str] = Field(
        description="List of valid cuisine types (e.g. 'american', 'italian', 'chinese')",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "diet_labels": ["balanced", "high-protein"],
                "health_labels": ["egg-free", "gluten-free"],
                "meal_types": ["breakfast", "lunch"],
                "dish_types": ["salad", "main course"],
                "cuisine_types": ["american", "italian"]
            }
        }

class ParameterStats(BaseModel):
    """Statistics for parameter success rates"""
    success: int = Field(default=0, description="Number of successful queries")
    total: int = Field(default=0, description="Total number of queries")
    


class RecipeDataCollector:
    def __init__(self,
                 params: RecipeParameters,
                 api_client: Any,
                 min_recipes_per_category: int = 100,
                 max_retries: int = 3,
                 rate_limit: float = 0.5):
        self.params = params
        self.api_client = api_client
        self.min_recipes = min_recipes_per_category
        self.max_retries = max_retries
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
                # Generate parameter combination
                params = self._generate_parameter_combination()
                print(f"Trying parameters: {params}")
                
                # Rate limiting
                time.sleep(self.rate_limit)
                
                # Get recipes
                recipes = await self.api_client.get_recipes(**params)
                
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
    
    
# We already have our RecipeParameters and ParameterStats classes defined

# Let's create a simple API client for Edamam
class EdamamClient:
    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key
        
    async def get_recipes(self, **params) -> list[dict]:
        """Get recipes from Edamam API"""
        base_params = {
            'type': 'public',
            'app_id': self.app_id,
            'app_key': self.app_key,
            'imageSize': 'THUMBNAIL',
            'random': 'true'
        }
        
        # Merge base params with provided params
        params = {**base_params, **params}
        
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.edamam.com/api/recipes/v2', params=params) as response:
                data = await response.json()
                return [hit['recipe'] for hit in data.get('hits', [])]

# Now let's use it:
async def main():
    # Initialize parameters with our lists from earlier
    params = RecipeParameters(
        diet_labels=diet_labels,
        health_labels=health_labels,
        meal_types=meal_types,
        dish_types=dish_types,
        cuisine_types=cuisine_types
    )

    # Create API client
    api_client = EdamamClient(
        app_id=edamam_api_id,
        app_key=edamam_api_key
    )

    # Create collector
    collector = RecipeDataCollector(
        params=params,
        api_client=api_client,
        min_recipes_per_category=10,  # Start small for testing
        max_retries=3,
        rate_limit=1.0  # 1 second between requests
    )

    # Collect recipes
    recipes = await collector.collect_recipes(target_recipes=100)  # Start with a small number
    
    # Print stats
    print(f"Collected {len(recipes)} recipes")
    print("\nParameter success rates:")
    for param_type, stats in collector.parameter_stats.items():
        success_rate = stats.success / stats.total if stats.total > 0 else 0
        print(f"{param_type}: {success_rate:.2%} ({stats.success}/{stats.total})")

    return recipes

# In a Python script, run it like this:
if __name__ == "__main__":
    asyncio.run(main())
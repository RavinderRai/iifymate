from pydantic import BaseModel, Field
from typing import Optional

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
    
class APIConfig(BaseModel):
    """API Configuration"""
    app_id: str
    app_key: str
    base_url: str = "https://api.edamam.com/api/recipes/v2"
    timeout: int = 30
    max_retries: int = 3
    
class CollectorConfig(BaseModel):
    """Recipe collection configuration"""
    target_recipes: int = 1000
    min_recipes_per_category: int = 100
    rate_limit: float = 10
    max_retries: int = 3
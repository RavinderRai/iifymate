from pydantic import BaseModel, Field
from ml_features.ml_calorie_estimation.src.databases.config import DatabaseConfig

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
    base_url: str
    timeout: int
    app_id: str    # Will be set from environment variable
    app_key: str   # Will be set from environment variable
    
class CollectorConfig(BaseModel):
    """Recipe collection configuration"""
    target_recipes: int
    min_recipes_per_category: int
    requests_per_minute: int
    max_retries: int
    
class Config(BaseModel):
    collection: CollectorConfig
    database: DatabaseConfig
    api: APIConfig
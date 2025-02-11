from typing import Optional
from pydantic import BaseModel

class MealAnalysisResult(BaseModel):
    """Pydantic model to store and validate the combined analysis results"""
    ingredients: str
    health_labels: str
    recipe_label: str
    combined_features: str
    raw_ingredient_response: Optional[dict] = None
    
    class Config:
        frozen = True
        extra = "forbid"
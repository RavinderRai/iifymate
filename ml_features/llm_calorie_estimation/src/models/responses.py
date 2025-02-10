from typing import List
from pydantic import BaseModel

class IngredientResponse(BaseModel):
    name: list[str]
    amount: list[str]
    unit: list[str]

class HealthLabelResponse(BaseModel):
    labels: List[str]

class RecipeNameResponse(BaseModel):
    name: str
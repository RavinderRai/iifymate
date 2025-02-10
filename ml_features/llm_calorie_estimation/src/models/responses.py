from typing import List
from pydantic import BaseModel

class IngredientResponse(BaseModel):
    name: list[str]
    amount: list[str]
    unit: list[str]

class HealthLabelResponse(BaseModel):
    labels: str

class RecipeLabelResponse(BaseModel):
    name: str
from pydantic import BaseModel

class IngredientResponse(BaseModel):
    ingredients: list[str]

class HealthLabelResponse(BaseModel):
    health_label: str

class RecipeLabelResponse(BaseModel):
    recipe_label: str
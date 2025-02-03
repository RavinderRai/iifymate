from sqlalchemy import Column, Integer, String, JSON, Float, ForeignKey
from sqlalchemy.orm import relationship
from ..base import BaseTable
    
class CleanRecipe(BaseTable):
    __tablename__ = 'clean_recipes'
    
    id = Column(Integer, primary_key=True)
    raw_recipe_id = Column(Integer, ForeignKey('raw_recipes.id'))
    label = Column(String)
    serving_size = Column(Integer)
    dietLabels = Column(JSON)
    healthLabels = Column(JSON)
    ingredientLines = Column(JSON)
    ingredients = Column(JSON)
    calories = Column(Float)
    totalWeight = Column(Float)
    totalTime = Column(Integer)
    cuisineType = Column(JSON)
    mealType = Column(JSON)
    dishType = Column(JSON)
    totalNutrients = Column(JSON)
    tags = Column(JSON)
    raw_recipe = relationship("RawRecipe")
    
    @classmethod
    def from_dict(cls, recipe_data: dict) -> 'CleanRecipe':
        """Create a Clean_Recipe instance from a dictionary"""
        return cls(
            raw_recipe_id=recipe_data['raw_recipe_id'],
            label=recipe_data.get('label', ''),
            serving_size=recipe_data.get('serving_size', 0.0),
            dietLabels=recipe_data.get('dietLabels', []),
            healthLabels=recipe_data.get('healthLabels', []),
            ingredientLines=recipe_data.get('ingredientLines', []),
            ingredients=recipe_data.get('ingredients', []),
            calories=recipe_data.get('calories', 0.0),
            totalWeight=recipe_data.get('totalWeight', 0.0),
            totalTime=recipe_data.get('totalTime', 0.0),
            cuisineType=recipe_data.get('cuisineType', []),
            mealType=recipe_data.get('mealType', []),
            dishType=recipe_data.get('dishType', []),
            totalNutrients=recipe_data.get('totalNutrients', {}),
            tags=recipe_data.get('tags', [])
        )
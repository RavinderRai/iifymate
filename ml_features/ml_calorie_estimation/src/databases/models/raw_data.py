from sqlalchemy import Column, Integer, String, JSON, Float
from ml_features.ml_calorie_estimation.src.databases.base import BaseTable

class RawRecipe(BaseTable):  # Changed to PascalCase for class naming convention
    __tablename__ = 'raw_recipes'
    
    id = Column(Integer, primary_key=True)
    uri = Column(String, unique=True)
    label = Column(String)
    url = Column(String)
    yield_ = Column(Integer)
    dietLabels = Column(JSON)
    healthLabels = Column(JSON)
    cautions = Column(JSON)
    ingredientLines = Column(JSON)
    ingredients = Column(JSON)
    calories = Column(Float)
    totalWeight = Column(Float)
    totalTime = Column(Integer)
    cuisineType = Column(JSON)
    mealType = Column(JSON)
    dishType = Column(JSON)
    totalNutrients = Column(JSON)
    totalDaily = Column(JSON)
    digest = Column(JSON)
    tags = Column(JSON)

    @classmethod
    def from_dict(cls, recipe_data: dict) -> 'RawRecipe':
        """Create a RawRecipe instance from a dictionary"""
        return cls(
            uri=recipe_data.get('uri', ''),
            label=recipe_data.get('label', ''),
            url=recipe_data.get('url', ''),
            yield_=recipe_data.get('yield', 0.0),
            dietLabels=recipe_data.get('dietLabels', []),
            healthLabels=recipe_data.get('healthLabels', []),
            cautions=recipe_data.get('cautions', []),
            ingredientLines=recipe_data.get('ingredientLines', []),
            ingredients=recipe_data.get('ingredients', []),
            calories=recipe_data.get('calories', 0.0),
            totalWeight=recipe_data.get('totalWeight', 0.0),
            totalTime=recipe_data.get('totalTime', 0.0),
            cuisineType=recipe_data.get('cuisineType', []),
            mealType=recipe_data.get('mealType', []),
            dishType=recipe_data.get('dishType', []),
            totalNutrients=recipe_data.get('totalNutrients', {}),
            totalDaily=recipe_data.get('totalDaily', {}),
            digest=recipe_data.get('digest', []),
            tags=recipe_data.get('tags', [])
        )


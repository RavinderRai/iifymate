import logging
from pathlib import Path

from ml_features.llm_calorie_estimation.src.extractors.ingredients import IngredientExtractor
from ml_features.llm_calorie_estimation.src.extractors.health_labels import HealthLabelExtractor
from ml_features.llm_calorie_estimation.src.extractors.recipe_name import RecipeLabelExtractor
from ml_features.llm_calorie_estimation.src.models.responses import IngredientResponse, HealthLabelResponse

logger = logging.getLogger(__name__)

class MealAnalyzer:
    """Service that coordinates extractors and formats ML input"""
    def __init__(
        self,
        api_key: str,
        ingredient_prompt: str,
        health_label_prompt: str,
        recipe_name_prompt: str
    ):
        self.ingredient_extractor = IngredientExtractor(api_key, ingredient_prompt)
        self.health_label_extractor = HealthLabelExtractor(api_key, health_label_prompt)
        self.recipe_name_extractor = RecipeLabelExtractor(api_key, recipe_name_prompt)
    
    def _format_ingredients(self, ingredients: IngredientResponse) -> str:
        """Convert ingredients to ML model format"""
        return ", ".join(
            f"{amount} {unit} {name}" 
            for name, amount, unit in zip(ingredients.name, ingredients.amount, ingredients.unit)
        )
    
    def _format_health_labels(self, health_labels: HealthLabelResponse) -> str:
        """Convert health labels to ML model format"""
        return ", ".join(health_labels.labels)
    
    def analyze_meal(self, image_path: str | Path) -> str:
        """Extract all information and format for ML model input"""
        try:
            ingredients = self.ingredient_extractor.extract(image_path)
            health_labels = self.health_label_extractor.extract(image_path)
            recipe_name = self.recipe_name_extractor.extract(image_path)
            
            formatted_ingredients = self._format_ingredients(ingredients)
            formatted_health_labels = self._format_health_labels(health_labels)
            
            return f"{formatted_health_labels} {recipe_name.name} {formatted_ingredients}"
        except Exception as e:
            logger.error(f"Error analyzing meal: {e}")
            raise
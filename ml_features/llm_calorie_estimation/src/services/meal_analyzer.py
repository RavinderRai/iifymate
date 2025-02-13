import logging
from pathlib import Path

from ml_features.llm_calorie_estimation.prompts.vision_prompts import INGREDIENT_LIST_PROMPT_TEMPLATE
from ml_features.llm_calorie_estimation.prompts.text_prompts import (
    HEALTH_LABEL_PROMPT_TEMPLATE,
    RECIPE_LABEL_PROMPT_TEMPLATE
)
from ml_features.llm_calorie_estimation.src.extractors.ingredients import IngredientExtractor
from ml_features.llm_calorie_estimation.src.extractors.health_labels import HealthLabelExtractor
from ml_features.llm_calorie_estimation.src.extractors.recipe_name import RecipeLabelExtractor
from ml_features.llm_calorie_estimation.src.models.meal_analysis import MealAnalysisResult
from ml_features.ml_calorie_estimation.src.feature_engineering.data_transformations import comma_to_bracket

logger = logging.getLogger(__name__)
        
class MealAnalyzer:
    def __init__(self, api_key: str):
        self.ingredient_extractor = IngredientExtractor(api_key=api_key)
        self.health_label_extractor = HealthLabelExtractor(api_key=api_key)
        self.recipe_label_extractor = RecipeLabelExtractor(api_key=api_key)
        
        logger.info("MealAnalyzer initialized successfully")
        
    def analyze_meal(self, image_path: str | Path) -> MealAnalysisResult:
        """
        Analyze a meal image to extract ingredients, health labels, and recipe labels.

        This function processes an image of a meal to identify its ingredients using
        a GPT-based model. It further extracts health labels and recipe labels from
        the identified ingredients. The function returns a structured result combining
        these extracted features.

        Args:
            image_path: Path to the meal image file.

        Returns:
            MealAnalysisResult: A Pydantic model containing processed ingredients,
            health labels, recipe label, combined features string suitable for ML model
            input, and the raw ingredient extraction response.
        """
        try:
            logger.info(f"Extracting ingredients from image: {image_path}")
            ingredient_response = self.ingredient_extractor.extract(
                image_path=image_path,
                prompt=INGREDIENT_LIST_PROMPT_TEMPLATE
            )
            
            logger.info("Extracting health label..")
            health_label_response = self.health_label_extractor.extract(
                input=ingredient_response.ingredients,
                prompt=HEALTH_LABEL_PROMPT_TEMPLATE
            )
            
            logger.info("Extracting recipe label..")
            recipe_label_response = self.recipe_label_extractor.extract(
                input=ingredient_response.ingredients,
                prompt=RECIPE_LABEL_PROMPT_TEMPLATE
            )
            
            processed_ingredients = comma_to_bracket(ingredient_response.ingredients)
            
            combined_features = f"{health_label_response.health_label} {recipe_label_response.recipe_label} {processed_ingredients}"

            logger.info("Successfully analyzed meal and prepared features")
            
            return MealAnalysisResult(
                ingredients=processed_ingredients,
                health_labels=health_label_response.health_label,
                recipe_label=recipe_label_response.recipe_label,
                combined_features=combined_features,
                raw_ingredient_response=ingredient_response.dict()
            )
        except Exception as e:
            logger.error(f"Error analyzing meal: {e}")
            raise
        
    def get_ml_features(self, image_path: str | Path) -> str:
        """
        Analyze a meal image and return only the combined features string for ML model input
        
        Args:
            image_path: Path to the meal image
            
        Returns:
            str: Combined features string ready for ML model
        """
        result = self.analyze_meal(image_path)
        return result.combined_features
        
    async def analyze_meal_async(self, image_path: str | Path) -> str:
        """
        Async version of get_ml_features for future use with async APIs
        
        This is a placeholder for future implementation when async APIs are needed
        Currently just calls the sync version
        """
        return self.get_ml_features(image_path)
    
if __name__ == "__main__":
    # To run and test this class, run the folling in the root directory:
    # python -m ml_features.llm_calorie_estimation.src.services.meal_analyzer
    import os
    from dotenv import load_dotenv
    
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    try:
        # Initialize analyzer
        analyzer = MealAnalyzer(api_key=openai_api_key)
        
        sample_img_path = "notebooks/data/sample_meal_images/scrambled_eggs.jpg"
        
        # Process image
        result = analyzer.analyze_meal(image_path=sample_img_path)
        
        # Print results in a readable format
        print("\nMeal Analysis Results:")
        print("-" * 50)
        print(f"Recipe Name: {result.recipe_label}")
        print(f"Health Labels: {result.health_labels}")
        print(f"Processed Ingredients: {result.ingredients}")
        print("\nCombined Features for ML Model:")
        print(result.combined_features)
        
    except FileNotFoundError:
        print("Error: Test image file not found. Please check the path.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")   
    
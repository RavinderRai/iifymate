import json
import base64
import logging
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI

from ml_features.llm_calorie_estimation.prompts.ingredients_prompt import INGREDIENT_LIST_PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngredientResponse(BaseModel):
    name: list[str]
    amount: list[str]
    unit: list[str]
    
class IngredientExtractor:
    def __init__(self, api_key: str, prompt: str = INGREDIENT_LIST_PROMPT_TEMPLATE):
        self.client = OpenAI(api_key=api_key)
        self.prompt = prompt
        
    def _encode_image(self, image_path: str | Path) -> str:
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def extract_ingredients(self, image_path: str | Path) -> dict:
        base64_image = self._encode_image(image_path)
        
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            response_format=IngredientResponse
        )
        
        return IngredientResponse(**json.loads(response.choices[0].message.content))
    
if __name__ == "__main__":
    # run this command in WSL in root directory to test:
    # python -m ml_features.llm_calorie_estimation.src.ingredient_extractor
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    sample_image_path = 'notebooks/data/sample_meal_images/chili-lime_chicken_bowl.jpg'
    
    extractor = IngredientExtractor(openai_api_key)
    ingredients = extractor.extract_ingredients(sample_image_path)

    for name, amount, unit in zip(ingredients.name, ingredients.amount, ingredients.unit):
        print(f"{amount} {unit} of {name}")
from pathlib import Path
from ml_features.llm_calorie_estimation.src.models.responses import IngredientResponse
from ml_features.llm_calorie_estimation.src.extractors.base import VisionExtractor

class IngredientExtractor(VisionExtractor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        
    def extract(self, image_path: str | Path, prompt: str) -> IngredientResponse:
        return self._get_gpt_response(image_path, prompt, IngredientResponse)
    
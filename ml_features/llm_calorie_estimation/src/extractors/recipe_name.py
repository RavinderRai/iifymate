from ml_features.llm_calorie_estimation.src.models.responses import RecipeLabelResponse
from ml_features.llm_calorie_estimation.src.extractors.base import TextExtractor

class RecipeLabelExtractor(TextExtractor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        
    def extract(self, input: list[str], prompt: str) -> RecipeLabelResponse:
        return self._get_gpt_response(input, prompt, RecipeLabelResponse)
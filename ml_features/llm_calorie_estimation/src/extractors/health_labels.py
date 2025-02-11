from ml_features.llm_calorie_estimation.src.models.responses import HealthLabelResponse
from ml_features.llm_calorie_estimation.src.extractors.base import TextExtractor

class HealthLabelExtractor(TextExtractor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        
    def extract(self, input: list[str], prompt: str) -> HealthLabelResponse:
        return self._get_gpt_response(input, prompt, HealthLabelResponse)
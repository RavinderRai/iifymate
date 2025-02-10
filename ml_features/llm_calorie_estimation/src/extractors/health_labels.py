from pathlib import Path
from ml_features.llm_calorie_estimation.src.models.responses import HealthLabelResponse
from ml_features.llm_calorie_estimation.src.extractors.base import VisionExtractor

class HealthLabelExtractor(VisionExtractor):
    def __init__(self, api_key: str, prompt: str):
        super().__init__(api_key)
        self.prompt = prompt
        
    def extract(self, image_path: str | Path) -> HealthLabelResponse:
        return self._get_vision_response(image_path, self.prompt, HealthLabelResponse)
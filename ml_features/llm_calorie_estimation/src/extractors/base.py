import logging
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel

from ml_features.llm_calorie_estimation.utils.config import load_yaml
from ml_features.llm_calorie_estimation.utils.image_utils import encode_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionExtractor:
    """Base class for vision-based extraction"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = load_yaml("ml_features/llm_calorie_estimation/config.yaml")["model"]["name"]
        
    def _get_vision_response(self, image_path: str | Path, prompt: str, response_format: BaseModel) -> dict:
        base64_image = encode_image(image_path)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            return response_format(**response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in vision response: {e}")
            raise
        
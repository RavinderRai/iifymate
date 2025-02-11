import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel
from typing import Union
from jinja2 import Template

from ml_features.llm_calorie_estimation.utils.config import load_yaml
from ml_features.llm_calorie_estimation.utils.image_utils import encode_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """Shared base functionality for all extractors"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = load_yaml("ml_features/llm_calorie_estimation/config.yaml")["model"]["name"]
        
    @abstractmethod
    def _get_gpt_response(self, input: Union[str, Path, list[str]], prompt: str, response_format: BaseModel) -> dict:
        """
        Get response from GPT model
        Args:
            input: Either image path for vision or text/ingredients for text-only LLM
            prompt: Base prompt template
            response_format: Pydantic model for response validation
        Returns:
            Parsed and validated response
        """
        pass

class VisionExtractor(BaseExtractor):        
    """Base class for vision-based extraction"""        
    def _get_gpt_response(self, input: Union[str, Path], prompt: str, response_format: BaseModel) -> dict:
        base64_image = encode_image(input)
        
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
            
            # Parse the JSON string into a dict first
            response_dict = json.loads(response.choices[0].message.content)
            
            # Then create the Pydantic model
            return response_format(**response_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {e}")
            logger.error(f"Raw response: {response.choices[0].message.content}")
            raise
        except Exception as e:
            logger.error(f"Error in vision response: {e}")
            raise
        
class TextExtractor(BaseExtractor):
    def _get_gpt_response(self, input: list[str], prompt: str, response_format: BaseModel) -> dict:
        """
        Get response from GPT model

        Args:
            input: List of strings for the ingredients
            prompt: Base prompt template
            response_format: Pydantic model for response validation

        Returns:
            Parsed and validated response
        """
        template = Template(prompt)
        
        formatted_prompt = template.render(ingredients=input)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt,
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON string into a dict first
            response_dict = json.loads(response.choices[0].message.content)
            
            # Then create the Pydantic model
            return response_format(**response_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {e}")
            logger.error(f"Raw response: {response.choices[0].message.content}")
            raise
        except Exception as e:
            logger.error(f"Error in text response: {e}")
            raise
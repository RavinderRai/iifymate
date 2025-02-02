from abc import ABC, abstractmethod
import aiohttp
from ml_features.ml_calorie_estimation.src.data_ingestion.models import APIConfig

class RecipeClient(ABC):
    """Abstract base class for recipe API clients"""
    @abstractmethod
    async def get_recipes(self, **params) -> list[dict]:
        pass
    
class EdamamClient(RecipeClient):
    def __init__(self, config: APIConfig):
        self.config = config
        
    async def get_recipes(self, **params) -> list[dict]:
        base_params = {
            'type': 'public',
            'app_id': self.config.app_id,
            'app_key': self.config.app_key,
            'imageSize': 'THUMBNAIL',
            'random': 'true'
        }
        params = {**base_params, **params}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.config.base_url, params=params) as response:
                data = await response.json()
                return [hit['recipe'] for hit in data.get('hits', [])]
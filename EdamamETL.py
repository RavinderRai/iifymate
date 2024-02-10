import requests
import json

# Access the API keys
with open('config.json', 'r') as f:
    config_data = json.load(f)

api_keys = config_data['api_keys']
APP_ID = api_keys['edamam_app_id']
APP_KEY = api_keys['edamam_app_key']
import requests
import json

# Access the API keys
with open('config.json', 'r') as f:
    config_data = json.load(f)

api_keys = config_data['api_keys']
APP_ID = api_keys['edamam_app_id']
APP_KEY = api_keys['edamam_app_key']

def recipe_search(ingredient, from_index=0, to_index=10):
    if to_index > 100:
        raise ValueError("to_index must be 100 at maximum")
    
    app_id = APP_ID  # Replace with your Edamam API app ID
    app_key = APP_KEY  # Replace with your Edamam API app key
    result = requests.get(
        'https://api.edamam.com/search?q={}&app_id={}&app_key={}&from={}&to={}'.format(
            ingredient, app_id, app_key, from_index, to_index
        )
    )
    data = result.json()
    return data['hits']

def get_recipe_df(recipes):
    recipes_lst = [recipes[i]['recipe'] for i in range(len(recipes))]
    return pd.DataFrame(recipes_lst)
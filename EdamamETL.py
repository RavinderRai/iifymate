import requests
import json
import pandas as pd

class RecipeETL:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.load_config()
        #self.db_conn = None
        
    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config_data = json.load(f)
        
        api_keys = self.config_data.get('api_keys', {})
        self.APP_ID = api_keys.get('edamam_app_id', None)
        self.APP_KEY = api_keys.get('edamam_app_key', None)

    def recipe_search(self, ingredient, from_index=0, to_index=100):
        if to_index > 100:
            self.logger.warning("to_index exceeds maximum value (100). Adjusting to 100.")
            to_index = 100
        
        result = requests.get(
            f'https://api.edamam.com/search?q={ingredient}&app_id={self.APP_ID}&app_key={self.APP_KEY}&from={from_index}&to={to_index}'
        )
        data = result.json()
        return data['hits']

    def get_recipe_df(self, recipes):
        recipes_lst = [recipe['recipe'] for recipe in recipes]
        return pd.DataFrame(recipes_lst)

    def save_to_csv(self, df, filename='recipes.csv'):
        df.to_csv(filename, index=False)

    """ implement this later - it will check if we are loading data into csv or database 
    def run_etl(self, ingredient):
        recipes = self.recipe_search(ingredient)
        df = self.get_recipe_df(recipes)
        
        # Choose the data loading method based on configuration
        if self.config_data.get('data_loading_method') == 'csv':
            self.save_to_csv(df)
        elif self.config_data.get('data_loading_method') == 'database':
            self.save_to_database(df)
        else:
            raise ValueError("Unsupported data loading method specified in config file")
    """

if __name__ == "__main__":
    etl = RecipeETL()
    ingredient = input("Enter the ingredient: ")
    #etl.run_etl(ingredient)
    #etl.close_database()  # Close database connection if open
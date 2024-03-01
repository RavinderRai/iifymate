import requests
import json
import os
import pandas as pd
import time
from google.cloud import bigquery
import pandas_gbq

class RecipeETL:
    """
    A class for performing ETL operations on recipe data from the Edamam API.

    Attributes:
        config_file (str): Path to the configuration file (default is 'config.json').
        config_data (dict): Loaded configuration data.
        APP_ID (str): Edamam API application ID.
        APP_KEY (str): Edamam API application key.

    Methods:
        __init__(self, config_file='config.json'): Initializes a RecipeETL object.
        load_config(self): Loads configuration data and extracts API keys.
        recipe_search(self, ingredient, from_index=0, to_index=100): Searches for recipes with Edamam API.
        get_recipe_df(self, recipes): Converts recipe data to DataFrame.
        save_to_csv(self, df, filename='recipes.csv'): Saves DataFrame to CSV.
        run_etl(self, ingredient): Performs ETL process (not implemented yet).
    """

    def __init__(self, config_file='config.json', gcp_config_file='flavourquasar-gcp-key.json'):
        """Initializes a RecipeETL object."""
        self.config_data = self.load_config(config_file_path)
        api_keys = self.config_data.get('api_keys', {})
        self.APP_ID = api_keys.get('edamam_app_id', None)
        self.APP_KEY = api_keys.get('edamam_app_key', None)

        self.gcp_config_file = gcp_config_file
        self.gcp_config_data = self.load_config(self.gcp_config_file)
        self.project_id = self.gcp_config_data.get('project_id', None)

"""
old function - can delete if newer one is working fine
    def load_config(self):
        #Loads configuration data and extracts API keys.
        with open(self.config_file, 'r') as f:
            self.config_data = json.load(f)
        
        api_keys = self.config_data.get('api_keys', {})
        self.APP_ID = api_keys.get('edamam_app_id', None)
        self.APP_KEY = api_keys.get('edamam_app_key', None)
"""

    def load_config(self, config_file_path):
        """
        Load configuration settings from a JSON file.
        Args:
            config_file_path (str): Path to the configuration file.
        Returns:
            dict: A dictionary containing the configuration settings.
        """
        try:
            with open(config_file_path, 'r') as file:
                config_data = json.load(file)
            return config_data
        except FileNotFoundError:
            print(f"Configuration file '{config_file_path}' not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from '{config_file_path}'. Make sure it's a valid JSON file.")
            return {}

    def recipe_search(self, ingredient, from_index=0, to_index=100):
        """
        Searches for recipes containing the specified ingredient using the Edamam API.
        Args:
            ingredient (str): The ingredient to search for in recipes.
            from_index (int, optional): The index of the first recipe to return (default is 0).
            to_index (int, optional): The index of the last recipe to return (default is 100 and cannot go past that).
    
        Returns:
            list: A list of recipe data retrieved from the Edamam API.
        """
        if to_index > 100:
            self.logger.warning("to_index exceeds maximum value (100). Adjusting to 100.")
            to_index = 100
        
        result = requests.get(
            f'https://api.edamam.com/search?q={ingredient}&app_id={self.APP_ID}&app_key={self.APP_KEY}&from={from_index}&to={to_index}'
        )
        data = result.json()
        return data['hits']

    def get_recipe_df(self, recipes):
        """Converts a list of recipe data to a DataFrame. Input should be the output of the recipe_search function."""
        recipes_lst = [recipe['recipe'] for recipe in recipes]
        return pd.DataFrame(recipes_lst)

    def save_to_csv(self, df, filename='recipes.csv'):
        """Saves a DataFrame to a CSV file."""
        df.to_csv(filename, index=False)

    def combine_recipe_data(self, df, column='commonIngredients', from_index=0, to_index=100):
        all_recipes = []
        
        # Iterate over each ingredient in the df
        for ingredient in df[column]:        
            
            # Make API request for recipes
            recipes = self.recipe_search(ingredient, from_index=from_index, to_index=to_index)
            
            # Convert the list of recipes to a dataframe
            ingredient_df = self.get_recipe_df(recipes)
            
            # Append the dataframe to the list of all recipes
            all_recipes.append(ingredient_df)
    
            # Wait for 6 seconds between requests to adhere to rate limit, i.e. 10 requests per minute
            time.sleep(6)
        
        # Combine all dataframes into one dataframe
        combined_df = pd.concat(all_recipes, ignore_index=True)
        
        return combined_df

    def run_extract(self, df, column='commonIngredients', from_index=0, to_index=100, filename='recipes.csv', data_loading_method='csv', dataset_id='edamam_recipes', table_id='edamam_raw_data'):
        """
        Uses previous functions to collect data an save it as a csv or load into a database.
        Args:
            df (Pandas DataFrame): A Dataframe with a list of ingredients.
            column (str, optional): The column name for the df with the ingredients. Defaults to commonIngredients.
            from_index (int, optional): The index of the first recipe to return (default is 0).
            to_index (int, optional): The index of the last recipe to return (default is 100 and cannot go past that).
            filename (str, optional): The name for the csv file that we will save the recipes dataframe under.
            data_loading_method (str, optional): Selects whether to save data as a csv file or load into a database. 
                                                 Takes values of 'csv' or 'database'. Defaults to csv.
            dataset_id (str, optional): The dataset_id for a bigquery dataset. Defaults to 'edamam_recipes'
            table_id (str, optional): The table_id for the recipe table in the dataset_id bigquery dataset. Defaults to 'edamam_raw_data'.
        Returns:
            list: A dataframe with the list of recipes.
        """
        recipes_df = self.combine_recipe_data(df, column=column, from_index=from_index, to_index=to_index)

        if data_loading_method =='csv':
            self.save_to_csv(recipes_df, filename=filename)
        elif data_loading_method =='database':
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp_config_file
            client = bigquery.Client(self.project_id)
            destination_table = f"{self.project_id}.{dataset_id}.{table_id}"
            pandas_gbq.to_gbq(df, destination_table, project_id=self.project_id, if_exists='append')
        else:
            raise ValueError("Unsupported data loading method specified in config file")
        return recipes_df

if __name__ == "__main__":
    etl = RecipeETL()
    ingredient = input("Enter the ingredient: ")
    #etl.run_extract(ingredient)
    #etl.close_database()  # Close database connection if open
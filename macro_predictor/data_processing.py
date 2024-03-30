import numpy as np
import pandas as pd
import json
import os
import ast
from google.cloud import bigquery
import pandas_gbq

def get_bigquery_data():
    #load data from bigquery
    gcp_config_file = '../flavourquasar-gcp-key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file

    with open(gcp_config_file, 'r') as file:
                    gcp_config_data = json.load(file)
    project_id = gcp_config_data.get('project_id', None)
        
    query = """
        SELECT healthLabels, label, ingredientLines, totalNutrients
        FROM `flavourquasar.edamam_recipes.edamam_raw_data`
    """
    raw_df = pandas_gbq.read_gbq(query, project_id=project_id)

    return raw_df

def comma_to_bracket(ingredient_list):
    """
    Input: ingredient_list (str): a list of strings, like ingredients of a recipe.
    Output: recipe (str): commas in individual elements from input string are removed, then they are all joined together with a comma, so commas seperate each ingredient now.
    """
    processed_ingredients = []
    for ingredient in ingredient_list:
        parts = ingredient.split(',', 1)  # Split at the first comma
        if len(parts) > 1:  # Check if there is a comma
            # Check if the part after the comma is already in brackets
            if '(' not in parts[1] and ')' not in parts[1]:
                parts[1] = f'({parts[1].strip()})'  # Put it in brackets
        processed_ingredients.append(' '.join(parts))

    # Join the processed strings with a comma and space now that we removed the commas in the individual strings
    recipe = ', '.join(processed_ingredients)

    return recipe

def replace_with_priority(labels):
    priority_order = ['Vegan', 'Vegetarian', 'Pescatarian', 'Paleo', 'Red-Meat-Free', 'Mediterranean']
    for label in priority_order:
        if label in labels:
            return label
    return 'Balanced'  # Handle case where no label matches priority_order, in which case the diet is balanced

def get_macros(nutrients_row):
    macros_dct = {}

    for nutrient in nutrients_row.keys():
        if nutrients_row[nutrient]['label'] == 'Fat':
            macros_dct['fat'] = nutrients_row[nutrient]['quantity']
        elif nutrients_row[nutrient]['label'] == 'Protein':
            macros_dct['protein'] = nutrients_row[nutrient]['quantity']
        elif nutrients_row[nutrient]['label'] == 'Carbohydrates (net)':
            macros_dct['carbs'] = nutrients_row[nutrient]['quantity']

    return macros_dct


if __name__ == "__main__":
    df = get_bigquery_data()

    recipe_name = df['label']

    ingredient_lines = df['ingredientLines'].apply(ast.literal_eval)
    ingredient_lines = ingredient_lines.apply(comma_to_bracket)

    priority_health_labels = df['healthLabels'].apply(ast.literal_eval)
    priority_health_labels = priority_health_labels.apply(replace_with_priority)

    nutrients = df['totalNutrients'].apply(ast.literal_eval)

    y = pd.DataFrame(list(nutrients.apply(lambda row: get_macros(row))))





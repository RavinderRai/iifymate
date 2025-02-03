"""
Data tranformation functions for feature engineering to prepare for model training.
"""
def comma_to_bracket(ingredient_list: list) -> str:
    """
    Input: ingredient_list (list): a list of strings, like ingredients of a recipe.
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

def replace_with_priority(labels: list) -> str:
    priority_order = ['Vegan', 'Vegetarian', 'Pescatarian', 'Paleo', 'Red-Meat-Free', 'Mediterranean']
    for label in priority_order:
        if label in labels:
            return label
    return 'Balanced'  # Handle case where no label matches priority_order, in which case the diet is balanced

def get_macros(nutrients_row: dict) -> dict:
    macros_dct = {}

    macros_dct['Fat'] = nutrients_row['Fat']['quantity']
    macros_dct['Protein'] = nutrients_row['Protein']['quantity']
    macros_dct['Carbohydrates (net)'] = nutrients_row['Carbohydrates (net)']['quantity']
    
    return macros_dct


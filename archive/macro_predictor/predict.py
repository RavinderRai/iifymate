import os
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from google.cloud import storage
import pandas_gbq
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
logging.basicConfig(level=logging.INFO)

def load_artifact_from_gcs(artifact_path, bucket_name='macro_predictor'):
    """
    Loads an artifact from a pickle file stored in Google Cloud Storage.
    Args:
        bucket_name (str): Name of the GCS bucket.
        artifact_path (str): Path within the bucket where the artifact is stored.
    Returns:
        object: Loaded pickle file.
    """
    # Initialize a client and bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Get the blob (pickle file) from GCS
    blob = bucket.blob(artifact_path)

    # Download the pickle file contents as bytes
    pickle_bytes = blob.download_as_string()

    # Load the dictionary from the downloaded pickle file contents
    loaded_artifact = pickle.loads(pickle_bytes)

    return loaded_artifact

def get_similar_ingredients(user_recipe_name, df, tfidf_fitted, recipe_name_col='label', ingredients_col='ingredientLines'):
    recipe_names_df = df[recipe_name_col]

    #if the recipe is a direct match with something in our database, then return that
    for idx in df.index:
        if user_recipe_name == recipe_names_df[idx]:
            return df['ingredientLines'][idx]
    
    #otherwise, take the cosine similarity, for which we need to preprocess all the text and perform tfidf first
    user_recipe_name = lemmatizing_reviews(remove_stop_words(user_recipe_name))
    user_recipe_name = tfidf_fitted.transform([user_recipe_name])

    recipe_names_tdidf = recipe_names_df.apply(remove_stop_words)
    recipe_names_tdidf = recipe_names_tdidf.apply(lemmatizing_reviews)
    recipe_names_tdidf = tfidf_fitted.transform(recipe_names_tdidf)

    #get the cosine similarities and take the max to get the most relevant recipe's index
    cosine_similarities = cosine_similarity(user_recipe_name, recipe_names_tdidf)
    most_similar_index = cosine_similarities.argmax()

    #use that index to return the ingredients list
    return df.loc[most_similar_index, ingredients_col]

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

def remove_stop_words(review):
    english_stop_words = stopwords.words('english')

    #get the words in the review as a list
    review_words = review.split()
    
    #make a new list with the same words but only if they are not a stop word
    removed_stop_words_list = [word for word in review_words if word not in english_stop_words]
    
    removed_stop_words = ' '.join(removed_stop_words_list)
    
    return removed_stop_words

def lemmatizing_reviews(review):
    lemmatizer = WordNetLemmatizer()

    #get review text as a list of words
    review_list = review.split()
    
    #lemmatize the words
    lemmatized_list = [lemmatizer.lemmatize(word) for word in review_list]
    
    #make it into a string again
    lemmatized_review = ' '.join(lemmatized_list)
    
    return lemmatized_review

if __name__ == "__main__":
    health_label_options = ['Vegan', 'Vegetarian', 'Pescatarian', 'Paleo', 'Red-Meat-Free', 'Mediterranean', 'Balanced']
    print("Select an option:")
    for i, option in enumerate(health_label_options, start=1):
        print(f"{i}. {option}")

    health_label_choice = input("Enter the health label: ")
    health_label_choice = int(health_label_choice)
    health_label = health_label_options[health_label_choice - 1]
    user_input_recipe = input("Enter the recipe name: ")

    gcp_config_file = '../flavourquasar-gcp-key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file

    with open(gcp_config_file, 'r') as file:
                    gcp_config_data = json.load(file)
    project_id = gcp_config_data.get('project_id', None)

    query = """
        SELECT healthLabels, label, ingredientLines, totalNutrients
        FROM `flavourquasar.edamam_recipes.edamam_raw_data`
    """

    df = pandas_gbq.read_gbq(query, project_id=project_id)

    svd_fitted = load_artifact_from_gcs('macro_data_processing/svd_fitted.pkl')
    tfidf_fitted = load_artifact_from_gcs('macro_data_processing/tfidf_fitted.pkl')

    XGBoost_fat_model = load_artifact_from_gcs('training/XGBoost_fat_model.pkl')
    XGBoost_carbs_model = load_artifact_from_gcs('training/XGBoost_carbs_model.pkl')
    XGBoost_protein_model = load_artifact_from_gcs('training/XGBoost_protein_model.pkl')

    #the user_input_recipe and health_label variables are input variables

    cosine_ingred = get_similar_ingredients(user_input_recipe, df, tfidf_fitted)

    full_user_input = health_label + ' ' + user_input_recipe + ' ' + comma_to_bracket(cosine_ingred)

    full_user_input = remove_stop_words(full_user_input)
    full_user_input = lemmatizing_reviews(full_user_input)
    full_user_input = tfidf_fitted.transform([full_user_input])
    full_user_input = svd_fitted.transform(full_user_input)

    predicted_fat = np.expm1(XGBoost_fat_model.predict(full_user_input)[0])
    predicted_carbs = np.expm1(XGBoost_carbs_model.predict(full_user_input)[0])
    predicted_protein = np.expm1(XGBoost_protein_model.predict(full_user_input)[0])

    calories = 9*predicted_fat + 4*(predicted_carbs + predicted_protein)

    print(user_input_recipe + f" has {int(predicted_carbs)} grams of carbs, {int(predicted_fat)} grams of fat, and {int(predicted_protein)} grams of protein")
    print(f"This totals to {int(calories)} calories")
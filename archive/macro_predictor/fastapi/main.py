from fastapi import FastAPI, HTTPException
import json
import pickle
import os
import ast
import pandas_gbq
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import storage
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

def remove_stop_words(review):
    #english_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
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
    """
    Retrieves a list of ingredients similar to the user input recipe name from a given dataset.
    Parameters:
        user_recipe_name (str): The name of the user input recipe to find ingredients for.
        df (DataFrame): The DataFrame containing recipe data, where each row represents a recipe.
        tfidf_fitted (object): The fitted TF-IDF vectorizer used initially from the data_processing.py file.
        recipe_name_col (str, optional): The name of the column in the DataFrame containing recipe names. Default is 'label'.
        ingredients_col (str, optional): The name of the column in the DataFrame containing ingredient lists. Default is 'ingredientLines'.

    Returns:
        list: A list of ingredients corresponding to the user input recipe name. If an exact match is found in the DataFrame, the ingredients of that recipe are returned. Otherwise, the ingredients of the most similar recipe based on name similarity using TF-IDF cosine similarity are returned.
    """
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

@app.post('/predict_ingredients')
async def predict_ingredients(user_input):
    user_input_recipe = user_input.get('user_input', None)
    if user_input_recipe is None:
        raise HTTPException(status_code=400, detail='Missing user_input')

    gcp_config_file = 'flavourquasar-gcp-key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file

    with open(gcp_config_file, 'r') as file:
        gcp_config_data = json.load(file)
    project_id = gcp_config_data.get('project_id', None)

    query = """
        SELECT label, ingredientLines
        FROM `flavourquasar.edamam_recipes.edamam_raw_data`
    """

    df = pandas_gbq.read_gbq(query, project_id=project_id)
    df['ingredientLines'] = df['ingredientLines'].apply(ast.literal_eval)

    tfidf_fitted = load_artifact_from_gcs('macro_data_processing/tfidf_fitted.pkl')
    cosine_ingred = get_similar_ingredients(user_input_recipe, df, tfidf_fitted)

    ingredients_lst = comma_to_bracket(cosine_ingred)
    ingredients_lst = [item.strip() for item in ingredients_lst.split(',')]

    return ingredients_lst

@app.post('/predict_macros')
async def predict_macros(user_input):
    user_input = user_input.get('user_input', None)
    if user_input is None:
        raise HTTPException(status_code=400, detail='Missing user_input')

    gcp_config_file = 'flavourquasar-gcp-key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file

    svd_fitted = load_artifact_from_gcs('macro_data_processing/svd_fitted.pkl')
    tfidf_fitted = load_artifact_from_gcs('macro_data_processing/tfidf_fitted.pkl')

    XGBoost_fat_model = load_artifact_from_gcs('training/XGBoost_fat_model.pkl')
    XGBoost_carbs_model = load_artifact_from_gcs('training/XGBoost_carbs_model.pkl')
    XGBoost_protein_model = load_artifact_from_gcs('training/XGBoost_protein_model.pkl')

    user_input = remove_stop_words(user_input)
    user_input = lemmatizing_reviews(user_input)
    user_input = tfidf_fitted.transform([user_input])
    user_input = svd_fitted.transform(user_input)

    predicted_fat = int(np.expm1(XGBoost_fat_model.predict(user_input)[0]))
    predicted_carbs = int(np.expm1(XGBoost_carbs_model.predict(user_input)[0]))
    predicted_protein = int(np.expm1(XGBoost_protein_model.predict(user_input)[0]))

    calories = 9*predicted_fat + 4*(predicted_carbs + predicted_protein)

    return {
        'predicted_fat': predicted_fat,
        'predicted_carbs': predicted_carbs,
        'predicted_protein': predicted_protein,
        'calories': calories
    }

@app.post('/test')
async def test(user_input: dict):
    user_input = user_input.get('user_input', None)
    if user_input is None:
        raise HTTPException(status_code=400, detail='Missing user_input')

    gcp_config_file = 'flavourquasar-gcp-key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file


    return {
        'user_input': user_input,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
import numpy as np
import pandas as pd
import json
import os
import ast
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas_gbq
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import aiplatform

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
logging.basicConfig(level=logging.INFO)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def get_bigquery_data():
    #load data from bigquery
    gcp_config_file = '../flavourquasar-gcp-key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file

    with open(gcp_config_file, 'r') as file:
                    gcp_config_data = json.load(file)
    project_id = gcp_config_data.get('project_id', None)
    client = bigquery.Client(project_id)
        
    query = """
        SELECT healthLabels, label, ingredientLines, totalNutrients
        FROM `flavourquasar.edamam_recipes.edamam_raw_data`
    """
    raw_df = pandas_gbq.read_gbq(query, project_id=project_id)

    return raw_df, client

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

def get_tfidf_splits(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    tfidf = TfidfVectorizer()

    tfidf_fitted = tfidf.fit(X_train.str.join(' '))

    tfidf_X_train_labels = tfidf_fitted.transform(X_train.str.join(' '))
    tfidf_X_test_labels = tfidf_fitted.transform(X_test.str.join(' '))
    tfidf_train_df = pd.DataFrame(tfidf_X_train_labels.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_test_df = pd.DataFrame(tfidf_X_test_labels.toarray(), columns=tfidf.get_feature_names_out())

    return tfidf_train_df, tfidf_test_df, y_train, y_test, tfidf_fitted

def SVD_reduction(X_train, X_test, n_components=1000):
    svd = TruncatedSVD(n_components=n_components)
    svd_fitted = svd.fit(X_train)
    X_train_reduced, X_test_reduced = svd.transform(X_train), svd.transform(X_test)

    #getting column names just to convert to dataframe
    column_names = [f"component_{i+1}" for i in range(X_train_reduced.shape[1])]
    X_train_reduced_df = pd.DataFrame(X_train_reduced, columns=column_names, index=X_train.index)
    X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=column_names, index=X_test.index)

    return X_train_reduced_df, X_test_reduced_df, svd_fitted

def delete_table(client, dataset_id, table_id):
    """
    Deletes a table in BigQuery.
    Args:
        client (BigQuery Client): Set client = bigquery.Client(project_id) before hand and use as input.
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The ID of the table to delete.
    Returns:
        None
    """
    table_ref = client.dataset(dataset_id).table(table_id)
    client.delete_table(table_ref, not_found_ok=True)

def upload_data_to_vertex_ai(client, df_train_test, display_name, table_id, dataset_id='macro_training_data', project_id='flavourquasar', delete_data=True):
    """
    Uploads training and testing data to a Vertex AI Dataset.

    Args:
        client (BigQuery Client): Set client = bigquery.Client(project_id) before hand and use as input.
        df_train_test (pandas DataFrame): DataFrame containing training and testing data.
        display_name (str): Display name for the Vertex AI Dataset.
        table_id (str): ID of the BigQuery table to create and store the data.
        dataset_id (str, optional): ID of the BigQuery dataset. Defaults to 'macro_training_data'.
        project_id (str, optional): Google Cloud project ID. Defaults to 'flavourquasar'.

    Returns:
        None
    """
    logging.info("Creating dataset in bigquery if it does not exists already...")
    bq_dataset_id = f"{project_id}.{dataset_id}"
    bq_dataset = bigquery.Dataset(bq_dataset_id)
    client.create_dataset(bq_dataset, exists_ok=True)

    # delete the table so it doesn't add the same data on top pre-existing data
    if delete_data:
        delete_table(client, dataset_id, table_id)

    logging.info("Creating table in bigquery if it does not exists already...")
    recipes_df_train_test = client.dataset(dataset_id).table(table_id)
    df_train_test_table = bigquery.Table(recipes_df_train_test)
    client.create_table(df_train_test_table, exists_ok=True)
    
    logging.info("Uploading data to Vertex AI...")
    aiplatform.TabularDataset.create_from_dataframe(
        df_source=df_train_test,
        staging_path=f"bq://{bq_dataset_id}.{table_id}",
        display_name=display_name,
    )

def create_bucket(bucket_name):
    """
    Creates a GCS bucket if it does not already exist.
    Args:
        bucket_name (str): The name of the GCS bucket to create.
    Returns:
        None
    """
    # Initialize the GCS client
    storage_client = storage.Client()

    # Check if the bucket already exists
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        # If the bucket does not exist, create it
        bucket.create()
        logging.info(f"Bucket {bucket.name} created.")
    else:
        logging.info(f"Bucket {bucket.name} already exists. Skipping creation.")

def upload_artifact_to_gcs(bucket_name, artifact_path, artifact):
    """
    Uploads a Python dictionary to Google Cloud Storage as a pickle file.
    Args:
        bucket_name (str): Name of the GCS bucket.
        artifact_path (str): Path within the bucket to store the artifact.
        dictionary (dict): Python dictionary to upload.
    Returns:
        None
    """
    # Convert artifact to a pickle byte stream
    pickle_data = pickle.dumps(artifact)

    # Initialize a client and bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload the pickle data to GCS
    blob = bucket.blob(artifact_path)
    blob.upload_from_string(pickle_data)

    logging.info(f"Artifact uploaded to: gs://{bucket_name}/{artifact_path}")

if __name__ == "__main__":
    df, client = get_bigquery_data()

    recipe_name = df['label']

    ingredient_lines = df['ingredientLines'].apply(ast.literal_eval)
    ingredient_lines = ingredient_lines.apply(comma_to_bracket)

    priority_health_labels = df['healthLabels'].apply(ast.literal_eval)
    priority_health_labels = priority_health_labels.apply(replace_with_priority)

    X = priority_health_labels + ' ' + recipe_name + ' ' + ingredient_lines
    X = X.rename('fullRecipeInput')
    X = X.apply(remove_stop_words)
    X = X.apply(lemmatizing_reviews)
    X = X.apply(lambda x: word_tokenize(x))

    nutrients = df['totalNutrients'].apply(ast.literal_eval)
    y = pd.DataFrame(list(nutrients.apply(lambda row: get_macros(row))))

    X_train, X_test, y_train, y_test, tfidf_fitted = get_tfidf_splits(X, y)

    
    X_train, X_test, svd_fitted = SVD_reduction(X_train, X_test, n_components=1000)

    y_train, y_test = np.log1p(y_train), np.log1p(y_test)

    upload_params_list = [
        {'df_train_test': X_train, 'display_name': 'X_train_table', 'table_id': 'X_train'},
        {'df_train_test': X_test, 'display_name': 'X_test_table', 'table_id': 'X_test'},
        {'df_train_test': y_train, 'display_name': 'y_train_table', 'table_id': 'y_train'},
        {'df_train_test': y_test, 'display_name': 'y_test_table', 'table_id': 'y_test'}
    ]

    for params in upload_params_list:
        upload_data_to_vertex_ai(client=client, **params)

    train_test_sets_shapes = {
            'X_train_shape': X_train.shape, 
            'X_test_shape': X_test.shape, 
            'y_train_shape': y_train.shape, 
            'y_test_shape': y_test.shape
        }
    
    bucket_name = 'macro_predictor'
    create_bucket(bucket_name)

    #create a list of dicts to iterate over for clarity
    artifacts = [
        {'path': 'macro_data_processing/train_test_sets_shapes.pkl', 'artifact': train_test_sets_shapes},
        {'path': 'macro_data_processing/svd_fitted.pkl', 'artifact': svd_fitted},
        {'path': 'macro_data_processing/tfidf_fitted.pkl', 'artifact': tfidf_fitted},
    ]

    for artifact in artifacts:
        upload_artifact_to_gcs(bucket_name, artifact['path'], artifact['artifact'])


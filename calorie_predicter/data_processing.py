"""
This script performs data preprocessing for a recipe dataset. 
It handles tasks such as target variable creation, preprocessing of dish and meal types,
feature engineering, and splitting the data into training and testing sets. 
It also saves artifacts such as models and encoders for future predictions.

"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import ast
import pickle
import json
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
import mlflow
import mlflow.sklearn
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import aiplatform
import pandas_gbq
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from calorie_predicter.utils import (
    round_up_to_nearest,
    filter_calories,
    sorted_binned_encoding,
    collapsing_to_priority,
    priority_list_dish_type,
    priority_list_meal_type,
    one_hot_encode,
    pre_process_text
)


def get_target_variable(df):
    """
    Create a new target variable by binning calories into intervals.
    Parameters:
        df (DataFrame): Input DataFrame containing recipe data.
    Returns:
        DataFrame: DataFrame with the target variable 'binnedCalories' appended.
    """
    calories_df = df['calories']

    # capping the calorie count, so we will include recipes with calorie counts so that we maintain 90% of our data
    filtered_calories_df = filter_calories(
        df, column='calories', quartile_percent=0.9)
    max_calorie_cutoff = round_up_to_nearest(max(filtered_calories_df))

    # binning the calorie count to turn this into a classification problem
    bin_edges = [i for i in range(0, int(max_calorie_cutoff)+1, 300)]
    labels = [
        f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges)-1)]
    binned_calories = pd.cut(
        filtered_calories_df, bins=bin_edges, labels=labels, include_lowest=True)
    binned_calories = binned_calories.rename('binnedCalories')

    # Assuming binned_calories is a pandas Series or DataFrame
    assert binned_calories.isna().sum(
    ) == 0, "The count of NaN values in binned_calories is not equal to 0"

    # sort and map the intervals to integers
    label_encoding = sorted_binned_encoding(binned_calories)
    target = binned_calories.map(label_encoding)

    # filter the original df with the indices of the new target to get new df
    binned_calories_df = df.loc[target.index]
    binned_calories_df = pd.concat([binned_calories_df, target], axis=1)

    return binned_calories_df


def preprocess_dish_type(pre_processed_df):
    """
    Preprocess the dish type column.
    Parameters:
        pre_processed_df (DataFrame): A DataFrame containing recipe data. 
                                      Typically the preprocessed output of get_target_variable
    Returns:
        DataFrame: DataFrame with 'dishTypeSkewedLabels' column added.
    """
    pre_processed_df = pre_processed_df.dropna(subset=['dishType'])
    dish_type_df = pre_processed_df['dishType'].apply(ast.literal_eval)
    dish_type_df = dish_type_df.rename('dishTypeLabel')

    # turning lists of values into just singular values so this will be a categorical column
    priority_list_dish_type_var = priority_list_dish_type()
    dish_type_df = dish_type_df.apply(
        lambda x: collapsing_to_priority(x, priority_list_dish_type_var))

    # putting them together
    pre_processed_df = pd.concat([pre_processed_df, dish_type_df], axis=1)

    # finding how skewed each category is
    skewness_by_category = pre_processed_df.groupby('dishTypeLabel')[
        'calories'].skew()

    # separating them into 3 skewedness by taking the min/max skewedness and binning them
    skewness_min = skewness_by_category.min()
    skewness_max = skewness_by_category.max()
    #now bin them
    interval_width = (skewness_max - skewness_min) / 3
    bin1_end = skewness_min + interval_width
    bin2_end = bin1_end + interval_width

    # just for clarity we will add a step to name these clearly, and then do label encoding right after
    bins = {
        'Left Skewed (Higher Calories)': skewness_by_category[(skewness_by_category >= skewness_min) & (skewness_by_category < bin1_end)],
        'Approximately Symmetric (Normal Calories)': skewness_by_category[(skewness_by_category >= bin1_end) & (skewness_by_category < bin2_end)],
        'Right Skewed (Lower Calories)': skewness_by_category[skewness_by_category >= bin2_end]
    }

    # flipping this around to create a map that converts each dish type label into a skewedness category
    skew_map = {}
    for skew in bins.keys():
        for category in bins[skew].index:
            skew_map[category] = skew

    pre_processed_df['dishTypeSkewedLabels'] = pre_processed_df['dishTypeLabel'].map(
        skew_map)

    # dish types with only 1 value will give nan values here, so we need to remove them
    pre_processed_df = pre_processed_df.dropna(subset=['dishTypeSkewedLabels'])
    pre_processed_df = pre_processed_df.reset_index(drop=True)

    # now we can quickly do label encoding for model training
    dish_type_map = {'Approximately Symmetric (Normal Calories)': 1,
                     'Right Skewed (Lower Calories)': 0, 'Left Skewed (Higher Calories)': 2}
    pre_processed_df['dishTypeSkewedLabels'] = pre_processed_df['dishTypeSkewedLabels'].map(
        dish_type_map)

    # return the maps to save and load in predict.py file
    return pre_processed_df, skew_map, dish_type_map


def preprocess_meal_type(pre_processed_df):
    """
    Preprocess the meal type column.
    Parameters:
        pre_processed_df (DataFrame): Preprocessed DataFrame containing recipe data.
                                      Typically the preprocessed output of preprocess_dish_type
    Returns:
        DataFrame: DataFrame with 'mealTypeRefined' column added.
    """
    meal_type_df = pre_processed_df['mealType'].apply(ast.literal_eval)

    # converting multilabel column into single label
    priority_list_meal_type_var = priority_list_meal_type()
    meal_type_df = meal_type_df.apply(
        lambda x: collapsing_to_priority(x, priority_list_meal_type_var))

    # replacing brunch and teatime with snack, effectively combining these categories
    replace_lst = ['brunch', 'teatime']
    replacement = 'snack'
    meal_type_df = meal_type_df.apply(
        lambda x: replacement if x in replace_lst else x)

    meal_type_df = meal_type_df.rename('mealTypeRefined')
    pre_processed_df = pd.concat([pre_processed_df, meal_type_df], axis=1)

    return pre_processed_df


def get_training_testing_data(df, X_columns, y_column, test_size=0.20, random_state=42):
    """
    Split data into training and testing sets and perform TF-IDF vectorization.
    Parameters:
        df (DataFrame): Input DataFrame containing recipe data.
        X_columns (list): List of feature columns.
        y_column (str): Name of the target column.
        test_size (float): Size of the testing set. Default is 0.20.
        random_state (int): Random seed for reproducibility. Default is 42.
    Returns:
        DataFrame: Training and testing sets for features and target variable.
        TfidfVectorizer: Fitted TF-IDF vectorizer.

    """
    X = df[X_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    tfidf = TfidfVectorizer()
    tfidf_fitted = tfidf.fit(X_train['label'].str.join(' '))

    tfidf_X_train_labels = tfidf.transform(X_train['label'].str.join(' '))
    tfidf_X_test_labels = tfidf.transform(X_test['label'].str.join(' '))

    tfidf_train_df = pd.DataFrame(
        tfidf_X_train_labels.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_test_df = pd.DataFrame(
        tfidf_X_test_labels.toarray(), columns=tfidf.get_feature_names_out())

    X_train_tfidf = pd.concat(
        [tfidf_train_df, X_train.drop('label', axis=1)], axis=1)
    X_test_tfidf = pd.concat(
        [tfidf_test_df, X_test.drop('label', axis=1)], axis=1)

    # return tfidf_fitted for mlflow tracking and we will need it for predicting on new inputs
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_fitted

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

def upload_data_to_vertex_ai(client, df_train_test, display_name, table_id, dataset_id='training_data', project_id='flavourquasar', delete_data=True):
    """
    Uploads training and testing data to a Vertex AI Dataset.

    Args:
        client (BigQuery Client): Set client = bigquery.Client(project_id) before hand and use as input.
        df_train_test (pandas DataFrame): DataFrame containing training and testing data.
        display_name (str): Display name for the Vertex AI Dataset.
        table_id (str): ID of the BigQuery table to create and store the data.
        dataset_id (str, optional): ID of the BigQuery dataset. Defaults to 'training_data'.
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
    dataset = aiplatform.TabularDataset.create_from_dataframe(
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
    mlflow.set_experiment("data_processing_experiment")
    experiment = mlflow.get_experiment_by_name("data_processing_experiment")

    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # Log script parameters
        mlflow.log_param('input_data_path', '../recipes.csv')
        mlflow.log_param('python_script', 'data_processing.py')

        # Load and preprocess raw data
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../flavourquasar-gcp-key.json"
        gcp_config_file = '../flavourquasar-gcp-key.json'
        with open(gcp_config_file, 'r') as file:
                gcp_config_data = json.load(file)
        project_id = gcp_config_data.get('project_id', None)
        
        client = bigquery.Client(project_id)
        
        query = """
            SELECT *
            FROM `flavourquasar.edamam_recipes.edamam_raw_data`
        """
        #raw_df = pandas_gbq.read_gbq(query, project_id=project_id)
        raw_df = pd.read_csv('../recipes.csv') #uncomment for faster loading if data is in a csv file locally
        df = raw_df.drop_duplicates('label')

        english_stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()

        # Data preprocessing steps
        pre_processed_df = get_target_variable(df)
        pre_processed_df, skew_map, dish_type_map = preprocess_dish_type(
            pre_processed_df)
        pre_processed_df = preprocess_meal_type(pre_processed_df)
        onehot_encoded_df, onehot_encoder = one_hot_encode(
            pre_processed_df, 'mealTypeRefined')
        pre_processed_df = pd.concat(
            [pre_processed_df, onehot_encoded_df], axis=1)

        pre_processed_df = pre_process_text(df=pre_processed_df,
                                            column='label',
                                            stop_words=english_stop_words,
                                            lemmatizer=lemmatizer,
                                            tokenizer=word_tokenize)

        # Convert columns to appropriate data types
        pre_processed_df['dishTypeSkewedLabels'] = pre_processed_df['dishTypeSkewedLabels'].astype(
            int)
        pre_processed_df['calorieLabels'] = pre_processed_df['binnedCalories'].astype(
            int)

        # Define features (X) and target variable (y) columns
        X_cols = ['mealTypeRefined_breakfast', 'mealTypeRefined_lunch/dinner',
                  'mealTypeRefined_snack', 'label', 'dishTypeSkewedLabels']
        y_col = 'binnedCalories'

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test, tfidf_fitted = get_training_testing_data(
            pre_processed_df, X_cols, y_col)

        # Now data and objects/artifacts are ready to be uploaded to GCP

        #bigquery won't accept column names with slashes in them
        X_train = X_train.rename({'mealTypeRefined_lunch/dinner': 'mealTypeRefined_lunch_dinner'}, axis=1)
        X_test = X_test.rename({'mealTypeRefined_lunch/dinner': 'mealTypeRefined_lunch_dinner'}, axis=1)
        y_train = y_train.to_frame()
        y_test = y_test.to_frame()

        upload_params_list = [
            {'df_train_test': X_train, 'display_name': 'X_train_table', 'table_id': 'X_train'},
            {'df_train_test': X_test, 'display_name': 'X_test_table', 'table_id': 'X_test'},
            {'df_train_test': y_train, 'display_name': 'y_train_table', 'table_id': 'y_train'},
            {'df_train_test': y_test, 'display_name': 'y_test_table', 'table_id': 'y_test'}
        ]
        
        # Iterate over the list and call upload_data_to_vertex_ai for each set of parameters
        for params in upload_params_list:
            upload_data_to_vertex_ai(client=client, **params)

        train_test_sets_shapes = {
            'X_train_shape': X_train.shape, 
            'X_test_shape': X_test.shape, 
            'y_train_shape': y_train.shape, 
            'y_test_shape': y_test.shape
        }

        bucket_name = 'calorie_predictor'
        create_bucket(bucket_name)

        #create a list of dicts to iterate over for clarity
        artifacts = [
            {'path': 'data_processing/train_test_sets_shapes.pkl', 'artifact': train_test_sets_shapes},
            {'path': 'data_processing/skew_map.pkl', 'artifact': skew_map},
            {'path': 'data_processing/dish_type_map.pkl', 'artifact': dish_type_map},
            {'path': 'data_processing/tfidf_fitted.pkl', 'artifact': tfidf_fitted},
            {'path': 'data_processing/onehot_encoder.pkl', 'artifact': onehot_encoder}
        ]

        for artifact in artifacts:
            upload_artifact_to_gcs(bucket_name, artifact['path'], artifact['artifact'])

        """
        # old code to save everythign locally. Uncomment if needed.
        X_train.to_csv('X_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)

        
        mlflow.log_param('X_train_shape', train_test_sets_shapes['X_train_shape'])
        mlflow.log_param('X_test_shape', train_test_sets_shapes['X_test_shape'])
        mlflow.log_param('y_train_shape', train_test_sets_shapes['y_train_shape'])
        mlflow.log_param('y_test_shape', train_test_sets_shapes['y_test_shape'])

        # saving tfidf, onehot encoder, and maps for predict.py file, to make predictions on unseen data
        with open("tfidf_model.pkl", "wb") as f:
            pickle.dump(tfidf_fitted, f)
        mlflow.log_artifact("tfidf_model.pkl", artifact_path=artifact_subdirectory)

        with open("skew_map.pkl", "wb") as f:
            pickle.dump(skew_map, f)
        mlflow.log_artifact("skew_map.pkl", artifact_path=artifact_subdirectory)

        # Save the dish_type_map dictionary
        with open("dish_type_map.pkl", "wb") as f:
            pickle.dump(dish_type_map, f)
        mlflow.log_artifact("dish_type_map.pkl", artifact_path=artifact_subdirectory)

        # Save the one-hot encoder
        with open("onehot_encoder.pkl", "wb") as f:
            pickle.dump(onehot_encoder, f)
        mlflow.log_artifact("onehot_encoder.pkl", artifact_path=artifact_subdirectory)
        """


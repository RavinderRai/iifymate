"""
train.py: Script for training an XGBoost model to predict each macronutrient.

This script loads training and testing data from VertexAI datasets, trains an XGBoost Regressor
with optimal parameters obtained from gridsearch testing earlier, and then saves the model in GCP.
"""
import json
import os
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from google.cloud import storage
from google.cloud import bigquery
import pickle
import logging
logging.basicConfig(level=logging.INFO)

def get_xgb_macro_model(X_train, X_test, y_train, y_test, macro, args):
    """
    Trains an XGBoost regressor model for predicting a specific macronutrient (carbs, fat, or protein) 
    using the recipe data from the Edamam API. Returns the trained model along with evaluation metrics.

    Parameters:
    X_train : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.
    X_test : array-like or sparse matrix, shape (n_samples, n_features)
        Test data.
    y_train : DataFrame, shape (n_samples, n_targets)
        Target values for training data.
    y_test : DataFrame, shape (n_samples, n_targets)
        Target values for test data.
    macro : str
        Name of the target macronutrient variable (column) in y_train and y_test.
    args : dict
        Dictionary containing arguments to be passed to the XGBRegressor constructor.

    Returns:
    xgb_model : XGBRegressor object
        Trained XGBoost regressor model.
    r2 : float
        R-squared score on the test data.
    mse : float
        Mean squared error on the test data.
    """
    xgb = XGBRegressor(**args)
    xgb.fit(X_train, y_train[macro])
    y_pred = xgb.predict(X_test)
    r2 = r2_score(y_test[macro], y_pred)
    mse = mean_squared_error(y_test[macro], y_pred)

    return xgb, r2, mse

def get_gcp_client(gcp_config_file = '../flavourquasar-gcp-key.json'):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file
    
    with open(gcp_config_file, 'r') as file:
            gcp_config_data = json.load(file)
    
    project_id = gcp_config_data.get('project_id', None)
    client = bigquery.Client(project_id)

    return client

def load_data_from_vertex_ai(client, table_id, dataset_id='macro_training_data', project_id='flavourquasar'):
    query = f"SELECT * FROM {project_id}.{dataset_id}.{table_id}"
    query_job = client.query(query)
    return query_job.to_dataframe()

def upload_artifact_to_gcs(artifact_path, artifact, bucket_name='macro_predictor'):
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
    client = get_gcp_client()

    X_train = load_data_from_vertex_ai(client=client, table_id='X_train')
    y_train = load_data_from_vertex_ai(client=client, table_id='y_train')
    X_test = load_data_from_vertex_ai(client=client, table_id='X_test')
    y_test = load_data_from_vertex_ai(client=client, table_id='y_test')

    #the carbs macro in y_train has some null values - forgot to deal with earlier
    #this is a temp fix, will need to return to this later
    y_train = y_train.dropna()
    X_train = X_train.loc[y_train.index]
    y_test = y_test.dropna()
    X_test = X_test.loc[y_test.index]

    fat_args = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': None}
    carbs_args = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': None}
    protein_args = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': None}

    logging.info("fitting XGBoost for fat macro..")
    fat_xgb, fat_r2, fat_mse = get_xgb_macro_model(X_train, X_test, y_train, y_test, 'fat', fat_args)
    
    logging.info("fitting XGBoost for carbs macro..")
    carbs_xgb, carbs_r2, carbs_mse = get_xgb_macro_model(X_train, X_test, y_train, y_test, 'carbs', carbs_args)
    
    logging.info("fitting XGBoost for protein macro..")
    protein_xgb, protein_r2, protein_mse = get_xgb_macro_model(X_train, X_test, y_train, y_test, 'protein', protein_args)

    #create dicts with meta data to track 
    metrics_dict = {
        'r2_test_score_fat': fat_r2,
        'mse_test_score_fat': fat_mse,
        'r2_test_score_carbs': carbs_r2,
        'mse_test_score_carbs': carbs_mse,
        'r2_test_score_protein': protein_r2,
        'mse_test_score_protein': protein_mse
    }

    xgb_args = {'fat': fat_args, 'carbs': carbs_args, 'protein': protein_args}

    upload_artifact_to_gcs('training/evaluation_metrics.pkl', metrics_dict)
    upload_artifact_to_gcs('training/xgb_args.pkl', xgb_args)
    upload_artifact_to_gcs('training/XGBoost_fat_model.pkl', fat_xgb)
    upload_artifact_to_gcs('training/XGBoost_carbs_model.pkl', carbs_xgb)
    upload_artifact_to_gcs('training/XGBoost_protein_model.pkl', protein_xgb)
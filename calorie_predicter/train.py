"""
train.py: Script for training an XGBoost model and logging it with MLflow.

This script loads training and testing data from CSV files, trains an XGBoost classifier
using grid search for hyperparameter tuning, evaluates the model, and logs the best model,
parameters, and metrics with MLflow.
"""
import os
import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
#import cupy as cp
from google.cloud import bigquery, aiplatform, storage
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import logging
logging.basicConfig(level=logging.INFO)

def load_data_from_vertex_ai(client, table_id, dataset_id='training_data', project_id='flavourquasar'):
    query = f"SELECT * FROM {project_id}.{dataset_id}.{table_id}"
    query_job = client.query(query)
    return query_job.to_dataframe()

def upload_artifact_to_gcs(artifact_path, artifact, bucket_name='calorie_predictor'):
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
    DEVICE_ID=0
    #cp.cuda.Device(DEVICE_ID).use()
    SEED = 42

    kappa_scorer = make_scorer(cohen_kappa_score)

    gcp_config_file = '../flavourquasar-gcp-key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config_file
    with open(gcp_config_file, 'r') as file:
            gcp_config_data = json.load(file)
    project_id = gcp_config_data.get('project_id', None)
    client = bigquery.Client(project_id)

    X_train = load_data_from_vertex_ai(client=client, table_id='X_train')
    y_train = load_data_from_vertex_ai(client=client, table_id='y_train')
    X_test = load_data_from_vertex_ai(client=client, table_id='X_test')
    y_test = load_data_from_vertex_ai(client=client, table_id='y_test')

    #X_train = pd.read_csv('X_train.csv')
    #y_train = pd.read_csv('y_train.csv')
    #X_test = pd.read_csv('X_test.csv')
    #y_test = pd.read_csv('y_test.csv')

    #X_train, y_train = cp.array(X_train.values), cp.array(y_train)
    #X_test, y_test = cp.array(X_test), cp.array(y_test)

    parameters = {
        'learning_rate': [0.1, 0.01, 0.001], 
        #'max_depth': [3, 5, 7],
        #'colsample_bytree': [0.6, 0.8, 1.0],
        #'n_estimators': [50, 100, 150]
    }

    xgb_clf = XGBClassifier(
                                objective='multi:softmax',
                                num_class=13,
                                random_state=SEED,
                                #device = "cuda"
                           )

    clf = GridSearchCV(xgb_clf, parameters, scoring='accuracy', n_jobs=-1)

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    best_params = clf.best_params_

    training_accuracy = accuracy_score(y_train_pred, y_train)
    testing_accuracy = accuracy_score(y_pred, y_test)
    training_kappa = cohen_kappa_score(y_train_pred, y_train)
    testing_kappa = cohen_kappa_score(y_pred, y_test)

    metrics_dict = {
        'training_accuracy': training_accuracy,
        'testing_accuracy': testing_accuracy,
        'training_kappa': training_kappa,
        'testing_kappa': testing_kappa
    }

    upload_artifact_to_gcs('training/evaluation_metrics.pkl', metrics_dict)
    upload_artifact_to_gcs('training/parameters.pkl', parameters)
    upload_artifact_to_gcs('training/XGBoost_model.pkl', clf.best_estimator_)

    """
    # uncomment if using mlflow locally
    mlflow.set_experiment("training_experiment")
    experiment = mlflow.get_experiment_by_name("training_experiment")

    # Log parameters, metrics, and model artifacts with MLflow
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Log grid search parameters
        for key, value in parameters.items():
            mlflow.log_param(f'grid_search_{key}', value)

        # Log metrics
        for metric_name, metric_value in metrics_dict.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log best parameters from grid search
        for key, value in clf.best_params_.items():
            mlflow.log_param(f'best_{key}', value)

        # Log XGBoost model
        mlflow.sklearn.log_model(clf.best_estimator_, 'xgboost_model')
    """

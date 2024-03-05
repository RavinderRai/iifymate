"""
Module to preprocess user input and output calorie predictions in a 300 calorie interval format.

Functions:
    - latest_run(directory): Retrieves the folder with the latest mlflow activity.
    - load_pickle(artifacts_path, file_name): Retrieves a pickle file in a given directory.
    - load_artifact_from_gcs(artifact_path, bucket_name='calorie_predictor'): Retrieves pickle file from GCP
    - preprocess_input(user_input, artifacts_path): Preprocesses user input data for the model.
    - post_process(num_of_classes): Creates a dictionary to map integers back to intervals.
    - upload_artifact_to_gcs(artifact_path, artifact, bucket_name='calorie_predictor'): Stores pickle file from GCP
"""
import os
import time
import pickle
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
from google.cloud import storage
#import mlflow
#import mlflow.sklearn
#from utils import get_experiment_folder_path

# Get a list of directories in mlruns/0 and then sort by creation time to get the latest one
def latest_run(directory):
    """
    Retrieves the folder with the latest mlflow activity within the specified directory.
    Parameters:
    - directory (str): The path to the directory containing multiple folders.
    Returns:
    - str: The path to the folder that was most recently created or updated.
    """
    run_directories = [d for d in os.listdir(
        directory) if os.path.isdir(os.path.join(directory, d))]
    latest_run_directory_id = max(
        run_directories, key=lambda d: os.path.getmtime(os.path.join(directory, d)))
    return directory + '/' + latest_run_directory_id


def load_pickle(artifacts_path, file_name):
    """
    Retrieves a pickle file in a given directory.
    Parameters:
    - artifacts_path (str): A directory path, typically an mlflow folder with artifacts.
    - file_name (str): The pickle file name.
    Returns:
    - object: The deserialized object from the pickle file.
    """
    with open(artifacts_path+file_name, "rb") as f:
        pickle_file = pickle.load(f)
        return pickle_file

def load_artifact_from_gcs(artifact_path, bucket_name='calorie_predictor'):
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


def preprocess_input(user_input):
    """
    Preprocesses user input data for model prediction.
    Parameters:
    - user_input (dict): Input data containing the following keys or columns:
        * 'label' (str): A recipe name of any kind. 
        * 'dishType' (str): The dish type category.
        * 'mealTypeRefined' (str): The meal type, options are: 'lunch/dinner', 'breakfast', 'snack'.
    - artifacts_path (str): The path to the directory containing the relevant mlflow artifacts.
    Returns:
    pandas.DataFrame: Preprocessed input data ready for model prediction.
    """

    #uncomment below if using mlflow and local files
    #tfidf_model = load_pickle(artifacts_path, "tfidf_model.pkl")
    #onehot_encoder = load_pickle(artifacts_path, "onehot_encoder.pkl")
    #skew_map = load_pickle(artifacts_path, "skew_map.pkl")
    #dish_type_map = load_pickle(artifacts_path, "dish_type_map.pkl")

    #loading preprocessing objects from GCP
    tfidf_model = load_artifact_from_gcs(artifact_path='data_processing/tfidf_fitted.pkl')
    onehot_encoder = load_artifact_from_gcs(artifact_path='data_processing/onehot_encoder.pkl')
    skew_map = load_artifact_from_gcs(artifact_path='data_processing/skew_map.pkl')
    dish_type_map = load_artifact_from_gcs(artifact_path='data_processing/dish_type_map.pkl')

    user_input = pd.DataFrame(user_input)

    user_input.loc[0, 'dishType'] = dish_type_map[skew_map[user_input['dishType'][0]]]
    user_input = user_input.rename({'dishType': 'dishTypeSkewedLabels'}, axis=1)

    column = 'mealTypeRefined'
    onehot_encoded_sample = onehot_encoder.transform(user_input[[column]])
    onehot_encoded_array = onehot_encoded_sample.toarray()
    onehot_encoded_df = pd.DataFrame(
        onehot_encoded_array,
        columns=onehot_encoder.get_feature_names_out([column])
    )

    user_input = pd.concat([user_input, onehot_encoded_df], axis=1)

    user_input = user_input[[
        'mealTypeRefined_breakfast',
        'mealTypeRefined_lunch/dinner', 'mealTypeRefined_snack',
        'label', 'dishTypeSkewedLabels'
    ]]

    
    # make sure this column is a float dtype and remove / from names since GCP doesn't take it
    user_input = user_input.rename({'mealTypeRefined_lunch/dinner': 'mealTypeRefined_lunch_dinner'}, axis=1)
    user_input['dishTypeSkewedLabels'] = user_input['dishTypeSkewedLabels'].astype(
        float)

    tfidf_user_input = tfidf_model.transform(user_input['label'])
    tfidf_user_input = pd.DataFrame(
        tfidf_user_input.toarray(), columns=tfidf_model.get_feature_names_out())

    tfidf_user_input_df = pd.concat(
        [tfidf_user_input, user_input.drop('label', axis=1)], axis=1)

    return tfidf_user_input_df

def post_process(num_of_classes):
    """
    Creates a dictionary to map integers to intervals.
    Parameters:
    - num_of_classes (int): The maximum integer to map to intervals.
    Returns:
    - dict: A dictionary where keys are ints and values are intervals of 300.
    The format appears as "start-end", i.e. "0-299", "300-500", and so on.
    """
    intervals = {}
    for i in range(num_of_classes + 1):
        start = i * 300
        end = (i + 1) * 300 - 1
        interval = f"{start}-{end}"
        intervals[i] = interval
    return intervals

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

def predict_calories(user_input, num_of_classes=13, artifact_path='training/XGBoost_model.pkl', log_artifacts=True, set_google_environment=True):
    if set_google_environment:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../flavourquasar-gcp-key.json"

    interval_map = post_process(num_of_classes=num_of_classes)
    loaded_model = load_artifact_from_gcs(artifact_path=artifact_path)

    # Make predictions and get probability estimates as well as latency
    start_time = time.time()
    class_probabilities = loaded_model.predict_proba(user_input)
    end_time = time.time()
    prediction_latency_ms = (end_time - start_time) * 1000  # milliseconds
    latency_data = {'latency': prediction_latency_ms, 'units': 'milliseconds'}

    #saving these to GCP
    if log_artifacts:
        upload_artifact_to_gcs('predicting/class_probabilities.pkl', class_probabilities)
        upload_artifact_to_gcs('predicting/model_latency.pkl', latency_data)

    #getting class from probabilities
    predicted_class_idx = class_probabilities.argmax(axis=1)[0]
    predicted_calorie_range = interval_map[predicted_class_idx]
    return predicted_calorie_range


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../flavourquasar-gcp-key.json"
    
    sample_user_input_raw = {
        'label': ['Carrot Soup'],
        'dishType': ['main course'],
        'mealTypeRefined': ['lunch/dinner']
    }

    sample_user_input = preprocess_input(sample_user_input_raw)

    # getting the max int to get the number of classes
    """
    y_train, y_test = pd.read_csv('y_train.csv'), pd.read_csv('y_test.csv')
    target_name = y_train.columns[0]
    max_class_int = max(
        list(y_train[target_name]) + list(y_test[target_name])
    )
    """

    predicted_calorie_range = predict_calories(
        sample_user_input, num_of_classes=13, 
        artifact_path='training/XGBoost_model.pkl', 
        log_artifacts=True, 
        set_google_environment=True)

    print(predicted_calorie_range, 'calories')

    """remove comments if using mlflow locally
    data_processing_experiment_id = get_experiment_folder_path('data_processing_experiment')
    mlflow_artifacts_path = latest_run(data_processing_experiment_id) + '/artifacts/'
    
    mlflow.set_experiment("predictions_experiment")
    experiment = mlflow.get_experiment_by_name("predictions_experiment")

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Log the input data
        mlflow.log_params(sample_user_input_raw)

        # Load the trained XGBoost model
        training_experiment_id = get_experiment_folder_path(
            'training_experiment')
        model_path = latest_run(training_experiment_id) + \
            '/artifacts/xgboost_model/'
        loaded_model = mlflow.sklearn.load_model(model_path)

        # Make predictions and get probability estimates as well as latency
        start_time = time.time()
        class_probabilities = loaded_model.predict_proba(sample_user_input)
        end_time = time.time()
        prediction_latency_ms = (end_time - start_time) * 1000  # milliseconds

        #getting class from probabilities
        predicted_class_idx = class_probabilities.argmax(axis=1)[0]
        predicted_calorie_range = interval_map[predicted_class_idx]
        print(predicted_calorie_range, 'calories')

        mlflow.log_metric('prediction_latency_ms', prediction_latency_ms)
        for class_idx, class_probabilities in enumerate(class_probabilities[0]):
            mlflow.log_metric(
                f'class_{class_idx}_probability', class_probabilities)

        mlflow.log_param('predicted_calorie_range', predicted_calorie_range)
    """

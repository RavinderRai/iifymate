"""
Module to preprocess user input and output calorie predictions in a 300 calorie interval format.

Functions:
    - latest_run(directory): Retrieves the folder with the latest mlflow activity.
    - load_pickle(artifacts_path, file_name): Retrieves a pickle file in a given directory.
    - preprocess_input(user_input, artifacts_path): Preprocesses user input data for the model.
    - post_process(num_of_classes): Creates a dictionary to map integers back to intervals.
"""

import os
import time
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from utils import get_experiment_folder_path

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


def preprocess_input(user_input, artifacts_path):
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
    tfidf_model = load_pickle(artifacts_path, "tfidf_model.pkl")
    onehot_encoder = load_pickle(artifacts_path, "onehot_encoder.pkl")
    skew_map = load_pickle(artifacts_path, "skew_map.pkl")
    dish_type_map = load_pickle(artifacts_path, "dish_type_map.pkl")

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
    # make sure this column is a float dtype
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

if __name__ == "__main__":
    data_processing_experiment_id = get_experiment_folder_path('data_processing_experiment')
    mlflow_artifacts_path = latest_run(data_processing_experiment_id) + '/artifacts/'

    # Load the test data
    X_test = pd.read_csv('X_test.csv')
    sample_test_input = X_test.iloc[0].to_numpy().reshape(1, -1)

    sample_user_input_raw = {
        'label': ['Carrot Soup'],
        'dishType': ['main course'],
        'mealTypeRefined': ['lunch/dinner']
    }

    sample_user_input = preprocess_input(sample_user_input_raw, mlflow_artifacts_path)

    # getting the max int to get the number of classes
    y_train, y_test = pd.read_csv('y_train.csv'), pd.read_csv('y_test.csv')
    target_name = y_train.columns[0]
    max_class_int = max(
        list(y_train[target_name]) + list(y_test[target_name])
    )

    interval_map = post_process(num_of_classes=max_class_int)

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

        # Make predictions and log probability estimates
        start_time = time.time()
        probabilities = loaded_model.predict_proba(sample_user_input)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000  # milliseconds

        mlflow.log_metric('prediction_latency_ms', latency_ms)
        for class_idx, class_probabilities in enumerate(probabilities[0]):
            mlflow.log_metric(
                f'class_{class_idx}_probability', class_probabilities)

        # Get the predicted class index and map it to the corresponding interval
        # calculate the max from probabilities so we don't run prediction twice
        predicted_class_idx = probabilities.argmax(axis=1)[0]
        predicted_calorie_range = interval_map[predicted_class_idx]
        print(predicted_calorie_range, 'calories')

        mlflow.log_param('predicted_calorie_range', predicted_calorie_range)

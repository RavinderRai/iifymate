import os
from pathlib import Path
from feast import FeatureStore
import pandas as pd
import mlflow
import logging

from ml_features.ml_calorie_estimation.src.training.multi_train import train_all_macro_models
from ml_features.ml_calorie_estimation.src.training.data_validation import clean_training_testing_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ML_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MLFLOW_TRACKING_URI = os.path.join(ML_PROJECT_ROOT, "mlruns")


def load_training_data():
    # Initialize feature store
    """
    Loads training and testing data from parquet files saved by the feature engineering pipeline
    and returns them as separate Pandas DataFrames.
    
    Returns:
        X_train (pd.DataFrame): The Pandas DataFrame containing the features for training.
        X_test (pd.DataFrame): The Pandas DataFrame containing the features for testing.
        y_train (pd.DataFrame): The Pandas DataFrame containing the targets for training.
        y_test (pd.DataFrame): The Pandas DataFrame containing the targets for testing.
    """
    store = FeatureStore("ml_features/ml_calorie_estimation/feature_store/feature_repo")
    
    # Load features and targets from parquet directly
    # Since we saved everything in one file, it's simpler to read directly
    feature_df = pd.read_parquet("ml_features/ml_calorie_estimation/feature_store/feature_repo/data/recipe_features.parquet")
    test_df = pd.read_parquet("ml_features/ml_calorie_estimation/feature_store/feature_repo/data/test_features.parquet")
    
    # Separate features and targets
    feature_cols = [col for col in feature_df.columns if col.startswith('component_')]
    target_cols = [col for col in feature_df.columns if col.startswith('target_')]
    
    X_train = feature_df[feature_cols]
    y_train = feature_df[target_cols]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]    

    return X_train, X_test, y_train, y_test

def run_training(env:str = "local"):
    """
    Trains a separate model for each macronutrient using the features saved in the feature store.
    
    Parameters:
    env (str): The environment to use for training. Options are 'local' or 'gcp' (default is 'local').
    
    Returns:
    None
    """
    X_train, X_test, y_train, y_test = load_training_data()
    
    macros = ['target_Fat', 'target_Carbohydrates_net', 'target_Protein']
    
    # Clean both train and test data
    logger.info("Cleaning training data...")
    X_train, y_train = clean_training_testing_data(X_train, y_train, macros)
    
    logger.info("Cleaning test data...")
    X_test, y_test = clean_training_testing_data(X_test, y_test, macros)
    
    
    # mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    # experiment_name = "macro_nutrient_prediction_dev" if env == "local" else "macro_nutrient_prediction_prod"
    
    # try:
    #     mlflow.create_experiment(
    #         experiment_name,
    #         artifact_location=f"file://{os.path.join(MLFLOW_TRACKING_URI, experiment_name)}"
    #     )
    # except Exception:
    #     pass  # Experiment already exists
    
    #mlflow.set_experiment(experiment_name)
    
    _, _ = train_all_macro_models(X_train, X_test, y_train, y_test, env)
    
    # for macro in ['target_Fat', 'target_Carbohydrates_net', 'target_Protein']:
    #     logging.info(f"\n{macro.upper()} Results:")
    #     logging.info(f"R2 Score: {metrics[macro]['r2']:.4f}")
    #     logging.info(f"MSE: {metrics[macro]['mse']:.4f}")
    #     logging.info("Best Parameters: %s", metrics[macro]['best_params'])

if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.pipeline.train
    
    run_training()
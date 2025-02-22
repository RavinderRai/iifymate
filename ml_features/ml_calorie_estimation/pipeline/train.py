import os
from pathlib import Path
import pandas as pd
import logging
from typing import Tuple

from ml_features.ml_calorie_estimation.src.data_ingestion.utils import load_config
from ml_features.ml_calorie_estimation.src.training.multi_train import train_all_macro_models
from ml_features.ml_calorie_estimation.src.training.data_validation import clean_training_testing_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ML_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MLFLOW_TRACKING_URI = os.path.join(ML_PROJECT_ROOT, "mlruns")

def get_feature_paths(env: str = "local") -> Tuple[str, str]:
    """Get the appropriate paths for feature files based on environment"""
    config = load_config(env)
    
    if env == "local":
        base_path = "ml_features/ml_calorie_estimation/feature_store/feature_repo/data"
        feature_path = f"{base_path}/recipe_features.parquet"
        test_path = f"{base_path}/test_features.parquet"
    else:
        bucket = config.aws.s3_bucket
        prefix = config.aws.feature_store_prefix
        feature_path = f"s3://{bucket}/{prefix}recipe_features.parquet"
        test_path = f"s3://{bucket}/{prefix}test_features.parquet"
    
    return feature_path, test_path

def load_training_data(env: str = "local") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    logger.info(f"Loading training data from {env} environment")
    feature_path, test_path = get_feature_paths(env)
    
    try:
        # Load features and targets
        logger.info(f"Reading training features from {feature_path}")
        feature_df = pd.read_parquet(feature_path)
        
        logger.info(f"Reading test features from {test_path}")
        test_df = pd.read_parquet(test_path)
        
        # Separate features and targets
        feature_cols = [col for col in feature_df.columns if col.startswith('component_')]
        target_cols = [col for col in feature_df.columns if col.startswith('target_')]
        
        if not feature_cols or not target_cols:
            raise ValueError("No feature or target columns found in the data")
        
        X_train = feature_df[feature_cols]
        y_train = feature_df[target_cols]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_cols]
        
        logger.info(f"Successfully loaded data: {len(X_train)} training samples, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def run_training(env:str = "local"):
    """
    Trains a separate model for each macronutrient using the features saved in the feature store.
    
    Parameters:
    env (str): The environment to use for training. Options are 'local' or 'gcp' (default is 'local').
    
    Returns:
    None
    """
    X_train, X_test, y_train, y_test = load_training_data(env)
    
    macros = ['target_Fat', 'target_Carbohydrates_net', 'target_Protein']
    
    # Clean both train and test data
    logger.info("Cleaning training data...")
    X_train, y_train = clean_training_testing_data(X_train, y_train, macros)
    
    logger.info("Cleaning test data...")
    X_test, y_test = clean_training_testing_data(X_test, y_test, macros)
    
    _, _ = train_all_macro_models(X_train, X_test, y_train, y_test, env)
    


if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.pipeline.train
    environment = os.getenv("ENV", "local")
    run_training(environment)
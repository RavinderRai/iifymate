import pandas as pd
import numpy as np
from typing import Tuple
import logging

from ml_features.ml_calorie_estimation.src.utils import load_config

logger = logging.getLogger(__name__)

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
        feature_path = f"s3://{bucket}/{prefix}recipe_features.csv"
        test_path = f"s3://{bucket}/{prefix}test_features.csv"
    
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
        feature_df = pd.read_csv(feature_path)
        
        logger.info(f"Reading test features from {test_path}")
        test_df = pd.read_csv(test_path)
        
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

def clean_training_testing_data(X: pd.DataFrame, y: pd.DataFrame, macros: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates and cleans data for XGBoost training by removing rows with NaN, inf values,
    or extremely large values in either features or all macro nutrients.
    
    IMPORTANT: Ideally this should be handled in the feature engineering pipeline, but for now, we're doing it here.
    
    Args:
        X: Feature DataFrame
        y: Target DataFrame with macro columns
        macros: List of macro column names to validate
        
    Returns:
        Cleaned X and y DataFrames
    """
    initial_rows = len(X)
    
    # Check for NaN or inf in features
    feature_mask = ~X.isna().any(axis=1) & ~np.isinf(X).any(axis=1)
    
    # Log feature issues if any
    nan_features = X.isna().sum()
    inf_features = np.isinf(X).sum()
    if nan_features.any() or inf_features.any():
        logger.warning("Features containing NaN or inf values:")
        for col in X.columns:
            if nan_features[col] > 0 or inf_features[col] > 0:
                logger.warning(f"{col}: {nan_features[col]} NaN, {inf_features[col]} inf")

    # Check for NaN or inf in all macro targets
    target_masks = []
    for macro in macros:
        macro_mask = ~y[macro].isna() & ~np.isinf(y[macro])
        target_masks.append(macro_mask)
        
        # Log target issues if any
        nan_count = y[macro].isna().sum()
        inf_count = np.isinf(y[macro]).sum()
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"Target {macro}: {nan_count} NaN, {inf_count} inf")

    # Combine all masks
    target_mask = pd.concat(target_masks, axis=1).all(axis=1)
    valid_mask = feature_mask & target_mask
    
    # Apply masks to both X and y
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    # Log cleaning results
    removed_rows = initial_rows - len(X_clean)
    if removed_rows > 0:
        logger.warning(
            f"Removed {removed_rows} rows ({(removed_rows/initial_rows)*100:.2f}%) "
            f"containing NaN or inf values"
        )
    
    return X_clean, y_clean
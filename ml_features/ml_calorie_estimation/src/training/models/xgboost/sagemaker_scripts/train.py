import os
import numpy as np
import argparse
import pandas as pd
import xgboost as xgb
import joblib
import logging

logger = logging.getLogger(__name__)

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

def load_training_data(data_path, target_column):
    """
    Loads the training dataset from an S3 path and extracts the required target column.
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)

    # Extract features and the specific target
    feature_cols = [col for col in df.columns if col.startswith("component_")]
    X_train = df[feature_cols]
    y_train = df[target_column]

    print(f"Loaded {X_train.shape[0]} samples with {len(feature_cols)} features.")
    return X_train, y_train

def train_model(X_train, y_train, args):
    """
    Trains an XGBoost model using the given hyperparameters.
    """
    params = {
        "objective": "reg:squarederror",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators
    }

    print(f"Training XGBoost model with params: {params}")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def main():
    parser = argparse.ArgumentParser()

    # SageMaker provides input/output directories as environment variables
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    # Hyperparameters passed from SageMaker training job
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--macro", type=str, required=True, help="Target column name")

    args = parser.parse_args()
    
    print(f"Training target column: {args.macro}")
    
    # Load data from S3
    X_train, y_train = load_training_data(args.train, args.macro)
    X_train, y_train = clean_training_testing_data(X_train, y_train, ['target_Fat', 'target_Carbohydrates_net', 'target_Protein'])

    # Train model
    model = train_model(X_train, y_train, args)

    # Save model to the specified output directory (SageMaker automatically uploads it to S3)
    model_path = os.path.join(args.model_dir, "xgboost-model.joblib")
    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()

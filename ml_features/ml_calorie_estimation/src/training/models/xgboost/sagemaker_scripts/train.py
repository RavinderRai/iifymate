import os
import json
import numpy as np
import argparse
import pandas as pd
import xgboost as xgb
import joblib
import logging
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_TRAIN_DIR = "/opt/ml/input/data/train"
INPUT_VALID_DIR = "/opt/ml/input/data/validation"
MODEL_DIR = "/opt/ml/model"

def custom_metric(y_true, y_pred):
    """Mock custom metric: r2_score + 1 (replace later with a real metric)."""
    custom_evaluation_metric = r2_score(y_true, y_pred) + 1
    return custom_evaluation_metric

def load_data(INPUT_DIR, macro):
    """Load training data from SageMaker input directory"""
    logger.info(f"Checking files in training directory: {INPUT_DIR}")

    # List all files in input directory
    files = os.listdir(INPUT_DIR)
    logger.info(f"Files found: {files}")

    # Check for unexpected file types (e.g., Parquet)
    if any(f.endswith(".parquet") for f in files):
        raise ValueError("🚨 Found Parquet files instead of CSV! Please check your S3 dataset.")

    # Find CSV files
    csv_files = [f for f in files if f.endswith(".csv")]
    if not csv_files:
        raise ValueError("❌ No CSV files found in training directory! Check if your dataset is correctly uploaded.")

    # Assume only one CSV file should be used for training
    data_path = os.path.join(INPUT_DIR, csv_files[0])
    logger.info(f"Using training data: {data_path}")

    # Try reading the file with UTF-8, fallback to ISO-8859-1 if needed
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
        logger.info("Successfully read CSV with UTF-8 encoding.")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed. Retrying with ISO-8859-1 encoding.")
        df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Print basic info about dataset
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {df.columns.tolist()}")
    logger.info(f"First 5 rows:\n{df.head()}")

    if macro not in df.columns:
        raise ValueError(f"Target column '{macro}' not found in dataset! Available columns: {df.columns}")

    X = df.drop(columns=[macro])  # Drop the dynamic macro target column
    y = df[macro]

    logger.info(f"Loaded training data with shape {df.shape} for target column: {macro}")
    return X, y

def train_model(X, y, hyperparameters):
    """Train XGBoost model"""
    logger.info(f"Training model with hyperparameters: {hyperparameters}")
    
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        **hyperparameters
    )
    model.fit(X, y)

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and log the metric for SageMaker"""
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    custom_value = custom_metric(y_test, y_pred)

    # 🚨 Handle potential NaN issue
    if np.isnan(r2):
        logger.warning("⚠️ r2_score is NaN! Setting to 0 to prevent SageMaker errors.")
        r2 = 0  # Set to a valid default to prevent failure
        custom_value = 1  # Since r2 + 1 would also be NaN

    # ✅ SageMaker expected format
    logger.info(f"validation:r2={r2}")  
    logger.info(f"custom_metric:r2_plus_1={custom_value}")

    # ✅ Save metrics in case SageMaker fails to capture them
    evaluation_output_path = os.path.join(MODEL_DIR, "evaluation_output.txt")
    with open(evaluation_output_path, "w") as f:
        json.dump({"validation:r2": r2, "custom_metric:r2_plus_1": custom_value}, f)

    return r2, custom_value

def main():
    # Parse SageMaker hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--macro", type=str, required=True)
    
    args = parser.parse_args()
    hyperparameters = vars(args)
    macro = args.macro

    # Load data
    X_train, y_train = load_data(INPUT_DIR=INPUT_TRAIN_DIR, macro=macro)
    X_test, y_test = load_data(INPUT_DIR=INPUT_VALID_DIR, macro=macro)

    # Train model
    model = train_model(X_train, y_train, hyperparameters)
    
    r2, custom_metric_value = evaluate_model(model, X_test, y_test)
    print(r2, custom_metric_value)

    # Save model (SageMaker expects it in `/opt/ml/model/`)
    model_path = os.path.join(MODEL_DIR, f"xgboost-{macro}-model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return r2, custom_metric_value

if __name__ == "__main__":
    main()

import argparse
import os
import pandas as pd
import xgboost as xgb
import mlflow

def parse_args():
    parser = argparse.ArguementParser()
    
    # Sagemaker specific arguments
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=100)
    
    return parser.parse_args()

def train():
    args = parse_args()
    
    train_data = pd.read_csv(f"{args.train}/train.csv")
    X = train_data.drop('target', axis=1)
    y = train_data['target']
    
    # Train model
    model = xgb.XGBRegressor(
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators
    )
    
    model.fit(X, y)
    
    # Save model
    model_path = os.path.join(args.model_dir, "model.json")
    model.save_model(model_path)
    
    # Log with MLflow
    mlflow.xgboost.log_model(model, "model")
    mlflow.log_params({
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators
    })
    
if __name__ == '__main__':
    train()
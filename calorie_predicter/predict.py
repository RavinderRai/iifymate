import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import os

mlruns_path = "mlruns/0"

# Get a list of directories in mlruns/0 and then sort by creation time to get the latest one
run_directories = [d for d in os.listdir(mlruns_path) if os.path.isdir(os.path.join(mlruns_path, d))]
latest_run_directory_id = max(run_directories, key=lambda d: os.path.getmtime(os.path.join(mlruns_path, d)))

# Load the MLflow model for the latest run
model_path = f"mlruns/0/{latest_run_directory_id}/artifacts/xgboost_model/"
loaded_model = mlflow.sklearn.load_model(model_path)

# Load the test data
X_test = pd.read_csv('X_test.csv')
sample_input = X_test.iloc[0].to_numpy().reshape(1, -1)

# Make predictions
predictions = loaded_model.predict(sample_input)

print(predictions)

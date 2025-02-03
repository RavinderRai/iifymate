import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
import mlflow.xgboost

def train_macro_model(X_train: pd.DataFrame, y_train: pd.DataFrame, macro: str, model_params: dict):
    """
    Trains an XGBoost model for a specific macronutrient with MLFlow tracking.
    """
    
    with mlflow.start_run(run_name=f"{macro}_training", nested=True):
        mlflow.log_params(model_params)
        
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train[macro])
        
        mlflow.xgboost.log_model(model, f"{macro}_model")
        
        return model
    
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame, macro: str) -> dict:
    """
    Evaluates the performance of an XGBoost model for a specific macronutrient.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test[macro], y_pred)
    mse = mean_squared_error(y_test[macro], y_pred)
    
    mlflow.log_metric(f"{macro}_r2", r2)
    mlflow.log_metric(f"{macro}_mse", mse)
    
    return {"r2": r2, "mse": mse}
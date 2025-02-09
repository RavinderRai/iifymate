import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

def train_macro_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    macro: str, 
    model_name: str,
    model_params: dict
):
    """
    Trains an XGBoost model for a specific macronutrient with MLFlow tracking.
    """
    with mlflow.start_run(run_name=f"{model_name}_training", nested=True):
        mlflow.set_tag("macro_type", macro)
        mlflow.set_tag("model_name", model_name)
        
        mlflow.log_params({f"{model_name}_{k}": v for k, v in model_params.items()})
        
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train[macro])
        
        # Create an input example using the first few rows of training data
        input_example = X_train.head(5)
        
        # Get model signature
        signature = infer_signature(X_train, y_train[macro])
        
        # Register model with signature and input example
        registered_model_name = f"xgboost_{model_name}"
        mlflow.xgboost.log_model(
            model, 
            f"{model_name}_model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )
        
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
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def grid_search_parameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv_folds: int = 3
) -> dict:
    """
    Performs grid search to find best parameters.
    Returns only the best parameters dict.
    """
    model = XGBRegressor()
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_
    
def train_macro_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    macro: str, 
    model_params: dict,
):
    """Grid search trains with a validation set, so we retrain on the whole training data using the best params"""
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train[macro])
    
    return model
    
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame, macro: str):
    """
    Evaluates the performance of an XGBoost model for a specific macronutrient.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test[macro], y_pred)
    mse = mean_squared_error(y_test[macro], y_pred)
        
    return {"r2": r2, "mse": mse}

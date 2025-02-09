import pandas as pd
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor

def grid_search_macro_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    macro: str,
    param_grid: dict,
    is_dev: bool = True
) -> dict:
    """
    Performs grid search with MLflow tracking
    """
    with mlflow.start_run(run_name=f"{macro}_grid_search", nested=True):
        # Log parameter grid
        mlflow.log_params({"param_grid": str(param_grid), "is_dev": is_dev})
        
        # Configure grid search based on environment
        if is_dev:
            # Use smaller subset for development
            subset_size = min(1000, len(X_train))
            X_train_subset = X_train[:subset_size]
            y_train_subset = y_train[:subset_size]
            cv_folds = 3
        else:
            X_train_subset = X_train
            y_train_subset = y_train
            cv_folds = 5
        
        # Initialize and run grid search
        model = XGBRegressor()
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1 if not is_dev else 2,
            verbose=1
        )
        
        grid_search.fit(X_train_subset, y_train_subset[macro])
        
        # Log best parameters and score
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score", grid_search.best_score_)
        
        return grid_search.best_params_

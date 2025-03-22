import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import logging

from ml_features.ml_calorie_estimation.src.training.models.model_base import ModelBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostModel(ModelBase):
    def __init__(self, env: str = "local"):
        super().__init__(env)
    
    def _train_macro_model(
        self,
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        macro: str, 
        model_params: dict,
    ) -> XGBRegressor:
        """Grid search trains with a validation set, so we retrain on the whole training data using the best params"""
        try:
            model = XGBRegressor(**model_params)
            
            logger.info(f"Training model for {macro}")
            model.fit(X_train, y_train[macro])
            
            return model
        except Exception as e:
            logger.error(f"Error training model for {macro}: {e}")
            raise e
    
    def _evaluate_macro_model(
        self, 
        model: XGBRegressor, 
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame, 
        macro: str
    ) -> dict[str, float]:
        """
        Evaluates the performance of an XGBoost model for a specific macronutrient.
        """
        logger.info(f"Making predictions on test set for {macro}")
        y_pred = model.predict(X_test)
        
        logger.info(f"Calculating metrics for {macro}")
        r2 = r2_score(y_test[macro], y_pred)
        mse = mean_squared_error(y_test[macro], y_pred)
            
        return {"r2": r2, "mse": mse}
        
    def _hyperparameter_tuning(
        self,
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
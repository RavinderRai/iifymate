import pandas as pd
from abc import ABC, abstractmethod

from xgboost import XGBRegressor

class ModelBase(ABC):
    def __init__(self, env: str = "local"):
        self.env = env        
    
    @abstractmethod
    def _train_macro_model(
        self,
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        macro: str, 
        model_params: dict,
    ) -> XGBRegressor:
        pass
    
    @abstractmethod
    def _evaluate_macro_model(
        self, 
        model: XGBRegressor, 
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame, 
        macro: str
    ) -> dict[str, float]:
        pass
    
    @abstractmethod
    def _hyperparameter_tuning(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: dict,
        cv_folds: int = 3
    ) -> dict:
        pass

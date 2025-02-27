import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

from xgboost import XGBRegressor

class ModelBase(ABC):
    def __init__(self, env: str = "local"):
        self.env = env        
    
    @abstractmethod
    def _train_macro_model(
        self,
        macro: str, 
        model_params: dict,
        X_train: Optional[pd.DataFrame] = None, 
        y_train: Optional[pd.Series] = None, 
    ) -> XGBRegressor:
        pass
    
    @abstractmethod
    def _evaluate_macro_model(
        self, 
        model: XGBRegressor, 
        macro: str,
        X_test: Optional[pd.DataFrame] = None, 
        y_test: Optional[pd.Series] = None, 
    ) -> dict[str, float]:
        pass
    
    @abstractmethod
    def _hyperparameter_tuning(
        param_grid: dict,
        cv_folds: int = 3,
        X_train: Optional[pd.DataFrame] = None, 
        y_train: Optional[pd.Series] = None, 
    ) -> dict:
        pass

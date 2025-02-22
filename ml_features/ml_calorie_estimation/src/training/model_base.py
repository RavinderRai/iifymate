from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
import mlflow
import boto3
import sagemaker
from sagemaker.xgboost import XGBoost
from pathlib import Path

class ModelBase(ABC):
    def __init__(self, model_name: str, env: str = "local"):
        self.model_name = model_name
        self.env = env
        self.model = None
        self.best_params = None
        
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        hyperparameters: Dict[str, Any]
    ) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model"""
        pass
    
    @abstractmethod
    def save(self, model_path: str) -> None:
        """Save model artifacts"""
        pass
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model artifacts"""
        pass
    
class XGBoostModel(ModelBase):
    def __init__(self, model_name, env = "local", config: Optional[Dict] = None):
        super().__init__(model_name, env)
        self.config = config or {}
        
        if env == "production":
            session = boto3.Session()
            self.sagamaker_session = sagemaker.Session()
            self.role = self.config.get("sagemaker_role")
            self.instance_type = self.config.get("instance_type", "ml.t2.medium")
            
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, hyperparameters: Dict[str, Any]) -> None:
        if self.env == "local":
            self._train_local(X_train, y_train, hyperparameters)
        elif self.env == "production":
            self._train_sagemaker(X_train, y_train, hyperparameters)
        else:
            raise ValueError(f"Unknown environment: {self.env}")
        
    def _train_local(self, X_train: pd.DataFrame, y_train: pd.DataFrame, hyperparameters: Dict[str, Any]) -> None:
        from xgboost import XGBRegressor
        
        self.model = XGBRegressor(**hyperparameters)
        self.model.fit(X_train, y_train[self.model_name])
        self.best_params = hyperparameters
        
    def _train_sagemaker(self, X_train: pd.DataFrame, y_train: pd.DataFrame, hyperparameters: Dict[str, Any]) -> None:
        train_path = f"s3://{self.config['bucket']}/'training_data'/{self.model_name}"
        
        training_input = sagemaker.inputs.TrainingInput(
            train_path, content_type='text/csv'
        )
        
        xgb = XGBoost(
            entry_point='train.py',
            source_dir = str(Path(__file__).parent / "sagemaker_scripts"),
            role=self.role,
            instance_type=self.instance_type,
            instance_count=1,
            framework_version='1.5-1',
            py_version='py3',
            hyperparameters=hyperparameters,
            output_path=f"s3://{self.config['bucket']}/'models'/{self.model_name}"
        )
        
        xgb.fit({"train": training_input})
        
        self.model = xgb
        self.best_params = hyperparameters
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.env == "local":
            return self.model.predict(X)
        elif self.env == "production":
            # Use SageMaker endpoint for prediction
            # Note we create the endpoint so getting the precition like this may take time
            predictor = self.model.deploy(
                initial_instance_count=1,
                instance_type=self.instance_type
            )
            predictions = predictor.predict(X.values)
            
            # Delete endpoint to prevent charges
            predictor.delete_endpoint()
            return predictions
        else:
            raise ValueError(f"Unknown environment: {self.env}")
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        from sklearn.metrics import r2_score, mean_squared_error
        
        y_pred = self.predict(X_test)
        
        return {
            "r2": r2_score(y_test[self.model_name], y_pred),
            "mse": mean_squared_error(y_test[self.model_name], y_pred)
        }
        
    def save(self, path: str) -> None:
        if self.env == "local":
            mlflow.xgboost.save_model(self.model, path)
        else:
            # Model is already saved in S3 by SageMaker
            pass
        
    def load(self, path: str) -> None:
        if self.env == "local":
            self.model = mlflow.xgboost.load_model(path)
        else:
            # Load from SageMaker model artifacts
            self.model = mlflow.pyfunc.load_model(path)
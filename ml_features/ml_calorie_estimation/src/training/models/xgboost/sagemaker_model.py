import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import logging
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter

from ml_features.ml_calorie_estimation.src.utils import load_config
from ml_features.ml_calorie_estimation.src.training.models.model_base import ModelBase
from ml_features.ml_calorie_estimation.src.training.data_loader import get_feature_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerModel(ModelBase):
    def __init__(self, env= "production"):
        super().__init__(env)
        
        config = load_config(env)
        aws_config = config.aws
        
        self.role = aws_config.sagemaker.role
        self.bucket = aws_config.s3_bucket
        self.instance_type = aws_config.sagemaker.instance_type
        self.instance_count = 1  # Keep at 1 for free tier
        self.output_path = aws_config.sagemaker.output_path
        self.sagemaker_session = sagemaker.Session()
        self.train_s3_uri, self.test_s3_uri = get_feature_paths(self.env)
        

    def _train_macro_model(self, macro, model_params, X_train=None, y_train=None):
        """
        Trains the final XGBoost model on SageMaker using the best hyperparameters.
        """
        model_params["macro"] = macro

        xgb_estimator = XGBoost(
            entry_point="train.py",
            source_dir="ml_features/ml_calorie_estimation/src/training/models/xgboost/sagemaker_scripts",
            framework_version="1.5-1",
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            output_path=self.output_path,
            sagemaker_session=self.sagemaker_session,
            hyperparameters=model_params
        )

        logger.info(f"Training final model for {macro} using data from {self.train_s3_uri}")
        xgb_estimator.fit({'train': TrainingInput(s3_data=self.train_s3_uri, content_type="csv")})
        self.latest_training_job = xgb_estimator.latest_training_job
        
        return xgb_estimator

    def _hyperparameter_tuning(self, macro, param_grid, max_jobs=2, max_parallel_jobs=1, X_train=None, y_train=None):
        """
        Uses SageMaker HyperparameterTuner to find the best hyperparameters.
        """
        logger.info(f"Starting hyperparameter tuning with parameter grid: {param_grid}")
        
        xgb_estimator = XGBoost(
            entry_point="train.py",
            source_dir="ml_features/ml_calorie_estimation/src/training/models/xgboost/sagemaker_scripts",
            framework_version="1.5-1",
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            output_path=self.output_path,
            sagemaker_session=self.sagemaker_session,
            hyperparameters={"macro": macro}
        )

        hyperparameter_ranges = {
            key: (IntegerParameter(*value) if isinstance(value[0], int) else ContinuousParameter(*value))
            for key, value in param_grid.items()
        }
        
        # Define Tuner
        tuner = HyperparameterTuner(
            estimator=xgb_estimator,
            objective_metric_name="validation:rmse",
            objective_type="Minimize",
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[{"Name": "validation:rmse", "Regex": "validation:rmse=([0-9\\.]+)"}],
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs
        )
        
        tuner.fit({
            'train': TrainingInput(s3_data=self.train_s3_uri, content_type="csv"),
            'validation': TrainingInput(s3_data=self.test_s3_uri, content_type="csv")
        })
        
        self.latest_tuning_job = tuner
        return tuner
    
    def _evaluate_macro_model(self, model, macro, X_test, y_test):
        pass
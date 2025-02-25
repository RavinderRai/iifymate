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
        

    def _train_macro_model(self, X_train, y_train, macro, model_params):
        """
        Trains the final XGBoost model on SageMaker using the best hyperparameters.
        """
        train_s3_uri, _ = get_feature_paths(self.env)
        
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

        logger.info(f"Training final model for {macro} using data from {train_s3_uri}")
        xgb_estimator.fit({'train': TrainingInput(s3_data=train_s3_uri, content_type="csv")})
        return xgb_estimator

    def _evaluate_macro_model(self, model, X_test, y_test, macro):
        """
        Evaluates the SageMaker-trained model by deploying a temporary endpoint.
        """
        predictor = model.deploy(initial_instance_count=1, instance_type=self.instance_type, endpoint_name=f"{macro}-endpoint")
        predictions = predictor.predict(X_test.values)

        self.sagemaker_session.delete_endpoint(predictor.endpoint)
        return {"r2": r2_score(y_test[macro], predictions), "mse": mean_squared_error(y_test[macro], predictions)}
    
    def _hyperparameter_tuning(self, X_train, y_train, macro, param_grid):
        """
        Uses SageMaker HyperparameterTuner to find the best hyperparameters.
        """
        train_s3_uri, _ = get_feature_paths(self.env)  # Use pre-stored S3 path
        
        xgb_estimator = XGBoost(
            entry_point="train.py",
            source_dir="ml_features/ml_calorie_estimation/src/training/models/xgboost/sagemaker_scripts",
            framework_version="1.5-1",
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            output_path=self.output_path,
            sagemaker_session=self.sagemaker_session,
            hyperparameters={"objective": "reg:squarederror", "num_round": 100, "macro": macro}
        )

        # Convert GridSearch param_grid to SageMaker format
        hyperparameter_ranges = {}
        for param, values in param_grid.items():
            if all(isinstance(v, int) for v in values):
                hyperparameter_ranges[param] = IntegerParameter(min(values), max(values))
            elif all(isinstance(v, float) for v in values):
                hyperparameter_ranges[param] = ContinuousParameter(min(values), max(values))
            else:
                hyperparameter_ranges[param] = CategoricalParameter(values)

        tuner = HyperparameterTuner(
            estimator=xgb_estimator,
            objective_metric_name="validation:rmse",
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=10,
            max_parallel_jobs=1,  # Free tier compatible
            objective_type="Minimize"
        )

        logger.info(f"Starting hyperparameter tuning for {macro} using data from {train_s3_uri}")
        tuner.fit({'train': TrainingInput(s3_data=train_s3_uri, content_type="csv")})
        tuner.wait()

        best_params = tuner.best_estimator().hyperparameters()
        logger.info(f"Best parameters for {macro}: {best_params}")
        return best_params
    
    def _evaluate_macro_model(self, model, X_test, y_test, macro):
        pass
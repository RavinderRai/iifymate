import os
import mlflow
from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFlowExperimentTracker:
    def __init__(self, model, macros: list, env: str = "local", aws_config: dict = None):
        self.model = model
        self.macros = macros
        self.env = env
        
        if self.env == "local":
            mlflow_dir = Path(__file__).parent.parent.parent / "mlruns"
            os.makedirs(mlflow_dir, exist_ok=True)
            mlflow_tracking_uri = f"file://{mlflow_dir}"
        else:
            # Use S3 for MLflow tracking
            mlflow_tracking_uri = f"s3://{aws_config.s3_bucket}/mlflow/"
            
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        experiment_name = f"macro_nutrient_prediction_{self.env}"
        try:
            mlflow.create_experiment(experiment_name)
        except Exception as e:
            logger.error(f"Error creating MLflow experiment: {e}")
            raise
        
        mlflow.set_experiment(experiment_name)
        
    def _track_hyperparameter_tuning(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame, 
        param_grid: dict
    ):
            logger.info("Training model with grid search...")
            best_params = self.model._hyperparameter_tuning(X_train, y_train, param_grid)
            mlflow.log_params(best_params)
            logger.info(f"Best parameters: {best_params}")
            
            return best_params
            
            
    def _track_train_macro_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        macro: str, 
        model_params: dict,
        model_name: str
    ):
        logger.info("Training final model with best params...")
        xgb_model = self.model._train_macro_model(X_train, y_train, macro, model_params)
        
        metrics = self.model._evaluate_macro_model(xgb_model, X_test, y_test, macro)
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")
        
        # Log model
        logger.info("Logging model...")
        signature = mlflow.models.infer_signature(
            X_train,
            xgb_model.predict(X_train)
        )
        input_example = X_train.head(5)
           
        mlflow.xgboost.log_model(
            xgb_model,
            model_name,
            signature=signature,
            input_example=input_example
        )
        
    def start_mlflow_run(
        self,
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        for macro in self.macros:  
            model_name = macro.lower().replace("target_", "")
            logger.info(f"\nTraining model for {macro}")
            
            if mlflow.active_run():
                mlflow.end_run()
                
            with mlflow.start_run(run_name=model_name) as run:
                logger.info(f"MLflow run ID: {run.info.run_id}")
                logger.info(f"Training {macro} model")
                
                # Set tags
                mlflow.set_tag("macro_type", macro)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("environment", self.env)
                
                best_params = self._track_hyperparameter_tuning(
                    X_train,
                    y_train,
                    self.model.param_grid
                )
                
                self._track_train_macro_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    macro=macro,
                    model_params=best_params,
                    model_name=model_name
                )
                logger.info(f"Run ID: {run.info.run_id}")

        mlflow.end_run()
            
import os
import tarfile
import boto3
import mlflow
from pathlib import Path
import logging
import datetime
import subprocess
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from ml_features.ml_calorie_estimation.src.utils import load_config

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
            #config = load_config(env)
            #db_config = config.database
            
            mlflow_tracking_uri = "postgresql://iifymateadmin:Quantum4ier!@iifymate-db.co5im862y9q7.us-east-1.rds.amazonaws.com/mlflowdb"
            
            # Use PostgreSQL-based MLFlow tracking
            #mlflow_tracking_uri = db_config.connection_string
            
            # Ensure MLFlow schema exists
            self._initialize_mlflow_db(mlflow_tracking_uri)
            
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"macro_nutrient_prediction_{self.env}_{experiment_timestamp}"
        try:
            mlflow.create_experiment(experiment_name)
        except Exception as e:
            logger.error(f"Error creating MLflow experiment: {e}")
            raise
        
        mlflow.set_experiment(experiment_name)
        
    def _initialize_mlflow_db(self, db_uri):
        logger.info("Checking if MLFlow database schema exists...")
        engine = create_engine(db_uri)

        try:
            with engine.connect() as conn:
                # ✅ Check if the `experiment` table exists (MLflow schema check)
                result = conn.execute(
                    text("SELECT 1 FROM information_schema.tables WHERE table_name = 'experiment'")
                ).fetchone()

                if result:
                    logger.info("MLflow database schema already exists. No upgrade needed.")
                else:
                    logger.info("MLflow schema not found. Running `mlflow db upgrade`...")
                    self._run_mlflow_db_upgrade(db_uri)
        
        except (OperationalError, ProgrammingError) as e:
            logger.error(f"Database connection error: {e}")
            logger.info("Attempting to initialize MLflow schema with `mlflow db upgrade`...")
            self._run_mlflow_db_upgrade(db_uri)
            
        finally:
            engine.dispose()
            logger.info("MLflow database schema check complete.")
        
    def _run_mlflow_db_upgrade(self, db_uri):
        """Run `mlflow db upgrade` command programatically"""
        try:
            logger.info("Running `mlflow db upgrade` command...")
            result = subprocess.run(
                ["mlflow", "db", "upgrade", db_uri],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"MLflow DB Upgrade Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running mlflow db upgrade: {e}")
            raise
        
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
        
    def _track_sagemaker_training(self, model_name, tuner=None):
        logger.info(f"Tracking SageMaker job for {model_name}")

        sm_client = boto3.client("sagemaker")

        # Get the best training job
        best_training_job = tuner.best_training_job()
        training_job_details = sm_client.describe_training_job(TrainingJobName=best_training_job)

        # Log hyperparameters
        best_params = training_job_details["HyperParameters"]
        mlflow.log_params(best_params)

        # Log best model's metrics
        training_metrics = training_job_details.get("FinalMetricDataList", [])
        logger.info(f"SageMaker Training Metrics: {training_metrics}")
        if not training_metrics:
            logger.warning(f"No training metrics found for {model_name}")
        
        seen_metrics = set()
        for metric in training_metrics:
            metric_name = metric["MetricName"]
            metric_value = metric["Value"]
            
            # ✅ Log metrics only once
            if metric_name not in seen_metrics:
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"Logged metric: {metric_name}={metric_value}")
                seen_metrics.add(metric_name)

            # Find the best model’s S3 location
            best_model_s3_uri = training_job_details["ModelArtifacts"]["S3ModelArtifacts"]
            logger.info(f"Best model URI: {best_model_s3_uri}")

        # Register the best model in MLflow
        mlflow.register_model(
            model_uri=best_model_s3_uri,
            name=f"xgboost_{model_name}"
        )

        logger.info(f"Registered best model for {model_name} in MLflow.")
        
        return best_params
    
    def _cleanup_s3_models(self, s3_bucket, keep_model):
        """Deletes all unnecessary models from the S3 bucket except the best one."""
        
        s3 = boto3.client("s3")
        objects = s3.list_objects_v2(Bucket=s3_bucket, Prefix="models/")

        if "Contents" in objects:
            for obj in objects["Contents"]:
                if obj["Key"] != keep_model:
                    logger.info(f"Deleting unused model: {obj['Key']}")
                    s3.delete_object(Bucket=s3_bucket, Key=obj["Key"])
        
    def start_mlflow_run(
        self,
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        param_grid: dict
    ):
        with mlflow.start_run(run_name="macro_nutrient_prediction") as parent_run:
            parent_run_id = parent_run.info.run_id
            
            for macro in self.macros:  
                model_name = macro.lower().replace("target_", "")
                logger.info(f"\nTraining model for {macro}")
                    
                with mlflow.start_run(run_name=model_name, nested=True) as run:
                    mlflow.set_tag("mlflow.parentRunId", parent_run_id)                    
                    mlflow.set_tag("macro_type", macro)
                    mlflow.set_tag("model_name", model_name)
                    mlflow.set_tag("environment", self.env)
                    
                    logger.info(f"MLflow run ID: {run.info.run_id}")
                    logger.info(f"Training {macro} model")
                    
                    if self.env == "local":
                                    
                        best_params = self._track_hyperparameter_tuning(
                            X_train,
                            y_train,
                            param_grid
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
                        
                    elif self.env == "production":
                        tuner = self.model._hyperparameter_tuning(
                            macro, 
                            param_grid
                        )
                        
                        self._track_sagemaker_training(
                            macro, 
                            tuner
                        )
                        
                    logger.info(f"Run ID: {run.info.run_id}")
                
                # Explicitly end the nested run
                mlflow.end_run()
                
        # Explicitly end the main run
        mlflow.end_run()

        logger.info("All runs completed. Closing MLflow run...")
        
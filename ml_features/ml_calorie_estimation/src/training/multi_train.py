import pandas as pd
import mlflow
from typing import Dict, Tuple
import logging
from pathlib import Path
import os
import shutil


from ml_features.ml_calorie_estimation.src.training.model_base import XGBoostModel
from ml_features.ml_calorie_estimation.src.training.model_utils import grid_search_parameters, train_macro_model, evaluate_model
from ml_features.ml_calorie_estimation.src.data_ingestion.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_mlruns():
    """Clean up mlruns directory if it exists"""
    mlflow_dir = Path(__file__).parent.parent.parent / "mlruns"
    if mlflow_dir.exists():
        shutil.rmtree(mlflow_dir)
    os.makedirs(mlflow_dir)

def setup_mlflow(env: str = "local", aws_config: Dict = None):
    """Set up MLflow tracking"""
    if env == "local":
        mlflow_dir = Path(__file__).parent.parent.parent / "mlruns"
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    else:
        # Use S3 for MLflow tracking
        mlflow.set_tracking_uri(f"s3://{aws_config.s3_bucket}/mlflow/")
    
    # Set up experiment
    experiment_name = f"macro_nutrient_prediction_{env}"
    try:
        mlflow.create_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Error creating MLflow experiment: {e}")
        raise
    
    mlflow.set_experiment(experiment_name)
    return experiment_name

def train_all_macro_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    env: str = "local",
) -> Tuple[Dict, Dict]:
    """
    Trains models for all macronutrients with grid search and MLflow tracking
    """
    # Clean up and set up MLflow
    if env == "local": 
        clean_mlruns()
        aws_config = None
    else:
        config = load_config(env)
        aws_config = config.aws

    experiment_name = setup_mlflow(env, aws_config)
    
    models = {}
    all_metrics = {}
    
    active_run = mlflow.active_run()
    if active_run:
        mlflow.end_run()
        
    param_grid = {
        'learning_rate': [0.01],
        'max_depth': [3, 5],
        'n_estimators': [100]
    }
    
    for macro in ['target_Fat', 'target_Carbohydrates_net', 'target_Protein']:
        model_name = macro.lower().replace("target_", "")
        logger.info(f"\nTraining model for {macro}")
        
        # Make sure no run is active before starting a new one
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name=model_name) as run:
            # Initialize model
            model = XGBoostModel(
                model_name=macro,
                env=env,
                config=config.aws if env == "production" else None
            )
            
            # Set tags
            mlflow.set_tag("macro_type", macro)
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("environment", env)
            
            # Train model
            model.train(X_train, y_train, param_grid)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            mlflow.log_metrics(metrics)
            
            # Log model
            if env == "local":
                model.save(f"models/{model_name}")
                mlflow.log_artifact(f"models/{model_name}")
            else:
                # Model is already saved in S3 by SageMaker
                mlflow.log_artifact(f"s3://{config.aws.bucket}/models/{model_name}")
            
            # Store results
            models[macro] = model
            all_metrics[macro] = {
                **metrics,
                'best_params': model.best_params,
                'run_id': run.info.run_id
            }
            
            logger.info(f"Run ID: {run.info.run_id}")
            logger.info(f"Best parameters: {model.best_params}")
            logger.info(f"Metrics: {metrics}")
            
            # logger.info(f"MLflow run ID: {run.info.run_id}")
            
            # # Set tags
            # mlflow.set_tag("macro_type", macro)
            # mlflow.set_tag("model_name", model_name)
            # mlflow.set_tag("environment", env)
            
            # # Find best parameters
            # logger.info("Finding best parameters...")
            # best_params = grid_search_parameters(X_train, y_train[macro], param_grid)
            # mlflow.log_params(best_params)
            
            # # Train model
            # logger.info("Training final model with best params...")
            # model = train_macro_model(X_train, y_train, macro, best_params)
            
            # # Evaluate
            # logger.info("Evaluating model...")
            # metrics = evaluate_model(model, X_test, y_test, macro)
            # mlflow.log_metrics(metrics)
            
            # # Log model
            # logger.info("Logging model...")
            # signature = mlflow.models.infer_signature(
            #     X_train,
            #     model.predict(X_train)
            # )
            # input_example = X_train.head(5)
            
            # mlflow.xgboost.log_model(
            #     model,
            #     model_name,
            #     signature=signature,
            #     input_example=input_example
            # )
            
            # # Store results
            # models[macro] = model
            # all_metrics[macro] = {
            #     **metrics,
            #     'best_params': best_params,
            #     'run_id': run.info.run_id
            # }
            
            # # Log results
            # logger.info(f"Run ID: {run.info.run_id}")
            # logger.info(f"Best parameters: {best_params}")
            # logger.info(f"Metrics: {metrics}")
    
    # Make sure we end the final run
    if mlflow.active_run():
        mlflow.end_run()
            
    return models, all_metrics
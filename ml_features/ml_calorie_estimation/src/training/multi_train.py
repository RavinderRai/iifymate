import mlflow
from typing import Dict, Tuple, Any
import logging

from ml_features.ml_calorie_estimation.src.training.model_utils import train_macro_model, evaluate_model
from ml_features.ml_calorie_estimation.src.training.grid_search import grid_search_macro_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_all_macro_models(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    is_dev: bool = True
) -> Tuple[Dict, Dict]:
    """
    Trains models for all macronutrients with grid search and MLflow tracking
    
    Returns:
        Tuple[Dict, Dict]: (models, metrics) dictionaries
    """
    experiment_name = "macro_nutrient_prediction_dev" if is_dev else "macro_nutrient_prediction_prod"
    mlflow.set_experiment(experiment_name)
    
    models = {}
    metrics = {}
    
    for macro in ['target_Fat', 'target_Carbohydrates_net', 'target_Protein']:
        # mlflow won't allow spaces in run names, and let's also remove the brackets
        model_name = macro.replace(" ", "_").replace("(", "").replace(")", "")
        logger.info(f"Training model for {model_name}")
        
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.01],
            'max_depth': [3, 5],
            'n_estimators': [None]
        }
        
        with mlflow.start_run(run_name=f"{macro}_full_pipeline") as parent_run:
            mlflow.set_tag("macro_type", macro)
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("environment", "development" if is_dev else "production")
            
            # Grid search
            logger.info(f"Starting grid search for {model_name}")
            best_params = grid_search_macro_model(
                X_train, y_train, macro, param_grid, is_dev=is_dev
            )
            
            # Train model with best parameters
            logger.info(f"Training final model for {model_name} with best parameters")
            model = train_macro_model(X_train, y_train, macro, model_name, best_params)
            
            # Evaluate model
            logger.info(f"Evaluating {macro} model")
            model_metrics = evaluate_model(model, X_test, y_test, macro)
            
            models[macro] = model
            metrics[macro] = {
                **model_metrics,
                'best_params': best_params
            }
            
            # Log artifacts
            mlflow.log_params({f"{model_name}_{k}": v for k, v in best_params.items()})
            mlflow.log_metrics({f"{model_name}_{k}": v for k, v in model_metrics.items()})
    
    return models, metrics

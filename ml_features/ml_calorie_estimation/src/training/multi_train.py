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
    mlflow.set_experiment("macro_nutrient_prediction")
    
    models = {}
    metrics = {}
    
    for macro in ['target_Fat', 'target_Carbohydrates (net)', 'target_Protein']:
        logger.info(f"Training model for {macro}")
        
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.01],
            'max_depth': [3, 5],
            'n_estimators': [None]
        }
        
        with mlflow.start_run(run_name=f"{macro}_full_pipeline") as parent_run:
            # Grid search
            logger.info(f"Starting grid search for {macro}")
            best_params = grid_search_macro_model(
                X_train, y_train, macro, param_grid, is_dev=is_dev
            )
            
            # Train model with best parameters
            logger.info(f"Training final model for {macro} with best parameters")
            model = train_macro_model(X_train, y_train, macro, best_params)
            
            # Evaluate model
            logger.info(f"Evaluating {macro} model")
            model_metrics = evaluate_model(model, X_test, y_test, macro)
            
            models[macro] = model
            metrics[macro] = {
                **model_metrics,
                'best_params': best_params
            }
            
            # Log artifacts
            mlflow.log_params(best_params)
            mlflow.log_metrics(model_metrics)
    
    return models, metrics

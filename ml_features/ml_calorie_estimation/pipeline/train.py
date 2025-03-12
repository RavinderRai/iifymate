import os
from pathlib import Path
import logging

from ml_features.ml_calorie_estimation.src.training.multi_train import train_all_macro_models
from ml_features.ml_calorie_estimation.src.training.data_loader import get_feature_paths, load_training_data, clean_training_testing_data
from ml_features.ml_calorie_estimation.src.training.models.xgboost.model import XGBoostModel
from ml_features.ml_calorie_estimation.src.training.models.xgboost.sagemaker_model import SageMakerModel
from ml_features.ml_calorie_estimation.src.training.experiment_tracker import MLFlowExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ML_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MLFLOW_TRACKING_URI = os.path.join(ML_PROJECT_ROOT, "mlruns")

def run_training(env:str = "local"):
    """
    Trains a separate model for each macronutrient using the features saved in the feature store.
    
    Parameters:
    env (str): The environment to use for training. Options are 'local' or 'gcp' (default is 'local').
    
    Returns:
    None
    """
    macros = ['target_Fat', 'target_Carbohydrates_net', 'target_Protein']
    
    param_grid = {
        'learning_rate': [0.01, 0.01],
        'max_depth': [3, 5],
        'n_estimators': [100]
    }
    X_train, X_test, y_train, y_test = load_training_data(env)
    
    if env == "local":
        
        # Clean both train and test data
        logger.info("Cleaning training data...")
        X_train, y_train = clean_training_testing_data(X_train, y_train, macros)
        
        logger.info("Cleaning test data...")
        X_test, y_test = clean_training_testing_data(X_test, y_test, macros)
        
        model_instance = XGBoostModel()
        experiment_tracker = MLFlowExperimentTracker(model_instance, macros, env)
        experiment_tracker.start_mlflow_run(
            X_train, 
            y_train, 
            X_test, 
            y_test,
            param_grid
        )
        
    elif env == "production":        
        model_instance = SageMakerModel(env)
        #xgb_model = model_instance._train_macro_model(X_train, y_train, macros[0], {"learning_rate": 0.01, "max_depth": 3, "n_estimators": 100})

        # SageMaker Tuning expects ranges as tuples, not lists of specific values
        param_grid = {
            'eta': (0.01, 0.2),  # SageMaker's version of learning_rate
            'max_depth': (3, 10),
            'num_round': (100, 500)  # SageMaker's version of n_estimators
        }
        
        experiment_tracker = MLFlowExperimentTracker(model_instance, macros, env)

        experiment_tracker.start_mlflow_run(None, None, None, None, param_grid)

        #tuner = model_instance._hyperparameter_tuning(macros[0], param_grid)
        
        


if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.pipeline.train
    environment = os.getenv("ENV", "local")
    run_training(environment)
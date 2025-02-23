import os
from pathlib import Path
import logging

from ml_features.ml_calorie_estimation.src.training.multi_train import train_all_macro_models
from ml_features.ml_calorie_estimation.src.training.data_loader import load_training_data, clean_training_testing_data

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
    X_train, X_test, y_train, y_test = load_training_data(env)
    
    macros = ['target_Fat', 'target_Carbohydrates_net', 'target_Protein']
    
    # Clean both train and test data
    logger.info("Cleaning training data...")
    X_train, y_train = clean_training_testing_data(X_train, y_train, macros)
    
    logger.info("Cleaning test data...")
    X_test, y_test = clean_training_testing_data(X_test, y_test, macros)
    
    _, _ = train_all_macro_models(X_train, X_test, y_train, y_test, env)
    


if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.pipeline.train
    environment = os.getenv("ENV", "local")
    run_training(environment)
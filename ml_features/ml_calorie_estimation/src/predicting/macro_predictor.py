import os
import numpy as np
import mlflow
import pandas as pd
import logging
from joblib import load
from pathlib import Path

from ml_features.ml_calorie_estimation.src.feature_engineering.text_processing import remove_stop_words, lemmatizing

logger = logging.getLogger(__name__)

class MacroPredictor:
    def __init__(self, env: str = "local"):
        
        # Set up paths for preprocessing models
        
        # This is for local testing
        #current_file = Path(__file__)
        #ml_calorie_path = current_file.parent.parent.parent  # This gets us to ml_calorie_estimation
        
        # This is for local testing with docker
        ml_calorie_path = Path("/app/ml_features/ml_calorie_estimation")
        
        # Construct paths for the joblib files
        feature_store_path = ml_calorie_path / "feature_store" / "feature_repo" / "data"
        tfidf_path = feature_store_path / "tfidf_fitted.joblib"
        svd_path = feature_store_path / "svd_fitted.joblib"
        
        # Load preprocessing models with error handling
        try:
            logger.info(f"Loading TF-IDF from {tfidf_path}")
            self.tfidf = load(tfidf_path)
            logger.info("TF-IDF loaded successfully")
            
            logger.info(f"Loading SVD from {svd_path}")
            self.svd = load(svd_path)
            logger.info("SVD loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Could not find preprocessing model file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading preprocessing models: {e}")
            raise
        
        # This is for local testing
        #mlflow_dir = Path(__file__).parent.parent.parent / "mlruns"
        
        # This is for local testing with docker
        mlflow_dir = ml_calorie_path / "mlruns"
        tracking_uri = str(mlflow_dir.absolute())
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(f"file://{tracking_uri}")
        
        # Get experiment name based on environment
        experiment_name = "macro_nutrient_prediction_dev" if env == "local" else "macro_nutrient_prediction_prod"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        
        logger.info("Loaded models from mlflow successfully")
        
        # Get run IDs for each model
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Initialize models dictionary
        self.models = {}
        
        # Map of macro types to their model names
        macro_types = {
            'fat': 'fat',
            'carbs': 'carbohydrates_net',
            'protein': 'protein'
        }
        
        for macro_type, model_name in macro_types.items():
            model_runs = runs[runs['tags.model_name'] == model_name]
            if len(model_runs) == 0:
                raise ValueError(f"No runs found for '{model_name}'.")
            
            latest_run = model_runs.iloc[0]
            run_id = latest_run.run_id
            
            # Construct model URI using container paths
            model_path = mlflow_dir / experiment.experiment_id / run_id / "artifacts" / model_name
            logger.info(f"Loading {macro_type} model from path: {model_path}")
            
            try:
                self.models[macro_type] = mlflow.xgboost.load_model(str(model_path))
                logger.info(f"Successfully loaded {macro_type} model")
            except Exception as e:
                logger.error(f"Error loading {macro_type} model: {e}")
                raise
            
            #model_uri = f"runs:/{run_id}/{model_name}"
            #self.models[macro_type] = mlflow.xgboost.load_model(model_uri)
            #logger.info(f"Loaded {macro_type} model from run: {run_id}")
            
        logger.info("Successfully loaded all models from MLflow")
            

    def preprocess_input(self, text: str) -> pd.DataFrame:
        text = remove_stop_words(text)
        text = lemmatizing(text)
        tfidf_vector = self.tfidf.transform([text])
        svd_features = self.svd.transform(tfidf_vector)
        return svd_features #pd.DataFrame(svd_features, columns=[f"component_{i}" for i in range(1, svd_features.shape[1]+1)])
    
    def predict(self, text_input) -> dict[str, float]:
        try:
            input = self.preprocess_input(text_input)
            predictions = {}
            
            for macro_type, model in self.models.items():
                pred = np.expm1(model.predict(input)[0])
                predictions[macro_type] = int(pred)
                
            calories = 9*predictions['fat'] + 4*predictions['protein'] + 4*predictions['carbs']
            predictions['calories'] = int(calories)
                
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
        
    def batch_predict(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.predict(text) for text in texts]
    
    
if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.src.predicting.macro_predictor    
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
    ML_PROJECT_ROOT = os.path.join(PROJECT_ROOT, "ml_calorie_estimation")
    MLFLOW_TRACKING_URI = os.path.join(ML_PROJECT_ROOT, "mlruns")

    predictor = MacroPredictor(
        MLFLOW_TRACKING_URI
    )

    sample_input = "Vegan Black Bean Tacos, 1/2 cups of black beans, 2 tortillas"

    predictions = predictor.predict(sample_input)
    print("Single Prediction:", sample_input)
    for macro, value in predictions.items():
        print(f"  {macro.capitalize()}: {value}")

    batch_inputs = ["Vegan Black Bean Tacos, 1/2 cups of black beans, 3 tortillas, 1/2 cup of salsa", "Balanced Black Bean Tacos, 1/2 cups of black beans, 2 tortillas"]
    batch_predictions = predictor.batch_predict(batch_inputs)
    
    print("\nBatch Predictions:")
    for i, prediction in enumerate(batch_predictions):
        print(f"  Input {i+1}: ", batch_inputs[i])
        for macro, value in prediction.items():
            print(f"    {macro.capitalize()}: {value}")
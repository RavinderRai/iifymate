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
    def __init__(self, mlflow_tracking_uri: str):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.tdidf = load("ml_features/ml_calorie_estimation/feature_store/feature_repo/data/tfidf_fitted.joblib")
        self.svd = load("ml_features/ml_calorie_estimation/feature_store/feature_repo/data/svd_fitted.joblib")
        
        self.models = {
            'fat': mlflow.xgboost.load_model("models:/xgboost_target_Fat/latest"),
            'carbs': mlflow.xgboost.load_model("models:/xgboost_target_Carbohydrates_net/latest"),
            'protein': mlflow.xgboost.load_model("models:/xgboost_target_Protein/latest")
        }
        logger.info("Loaded models from mlflow successfully")

    def preprocess_input(self, text: str) -> pd.DataFrame:
        text = remove_stop_words(text)
        text = lemmatizing(text)
        tfidf_vector = self.tdidf.transform([text])
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
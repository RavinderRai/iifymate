import os
import joblib
import numpy as np
import mlflow
import boto3
import pandas as pd
import logging
from joblib import load
from pathlib import Path

from ml_features.ml_calorie_estimation.src.feature_engineering.text_processing import remove_stop_words, lemmatizing

logger = logging.getLogger(__name__)

class MacroPredictor:
    def __init__(self, env: str = "local"):
        self.env = env
        self.models = {}
        self.s3_client = boto3.client("s3")
        
        # Map of macro types to their model names
        self.macro_types = {
            'fat': 'fat',
            'carbs': 'carbohydrates_net',
            'protein': 'protein'
        }
        
        self.experiment_name = "macro_nutrient_prediction_dev" if env == "local" else "macro_nutrient_prediction"
        
        # Set up paths for preprocessing models
        
        if self.env == "local":
            # This is for local testing
            #current_file = Path(__file__)
            #ml_calorie_path = current_file.parent.parent.parent  # This gets us to ml_calorie_estimation
            
            # This is for local testing with docker
            ml_calorie_path = Path("/app/ml_features/ml_calorie_estimation")
            
            # Construct paths for the joblib files
            feature_store_path = ml_calorie_path / "feature_store" / "feature_repo" / "data"
            self.tfidf_path = feature_store_path / "tfidf_fitted.joblib"
            self.svd_path = feature_store_path / "svd_fitted.joblib"
            
            self._load_local_transformers()
            
            # This is for local testing
            #mlflow_dir = Path(__file__).parent.parent.parent / "mlruns"
            
            # This is for local testing with docker
            self.mlflow_dir = ml_calorie_path / "mlruns"
            tracking_uri = str(self.mlflow_dir.absolute())
            logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
            mlflow.set_tracking_uri(f"file://{tracking_uri}")
            
            self._load_latest_local_models()
        else:
            tracking_uri = "postgresql://iifymateadmin:Quantum4ier!@iifymate-db.co5im862y9q7.us-east-1.rds.amazonaws.com/mlflowdb"
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set paths for transformers and models
            self.model_cache_dir = Path("/tmp/models")  # Cache directory inside the container
            self.transformer_dir = Path("/tmp/transformers")
            os.makedirs(self.model_cache_dir, exist_ok=True)
            os.makedirs(self.transformer_dir, exist_ok=True)
                        
            # Download latest transformers & models
            self._load_production_transformers()
            self._load_latest_production_models()
            
        logger.info("Successfully loaded all models from MLflow")
        

    
    def _load_latest_local_models(self):
        # Get experiment name based on environment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found.")
        
        logger.info("Loaded models from mlflow successfully")
        
        # Get run IDs for each model
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        for macro_type, model_name in self.macro_types.items():
            model_runs = runs[runs['tags.model_name'] == model_name]
            if len(model_runs) == 0:
                raise ValueError(f"No runs found for '{model_name}'.")
            
            latest_run = model_runs.iloc[0]
            run_id = latest_run.run_id
            
            # Construct model URI using container paths
            model_path = self.mlflow_dir / experiment.experiment_id / run_id / "artifacts" / model_name
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
        
    def _load_latest_production_models(self):
        for macro in self.macro_types:
            logger.info(f"Fetching latest model for {macro} from MLflow Registry...")
            
            #latest_version = max(
            #    [int(v.version) for v in mlflow.MlflowClient().get_registered_model(model_name).latest_versions]
            #)
            
            #model_uri = f"models:/{model_name}/{latest_version}"
            
    def _load_latest_production_models(self):
        """Fetch latest models from MLflow Registry & download from S3."""
        macro_types = ["fat", "carbohydrates_net", "protein"]
        
        for macro in macro_types:
            logger.info(f"Fetching latest model for {macro} from MLflow Registry...")
            
            # Get the latest model version from MLflow Registry
            model_version = mlflow.registered_model.get_latest_versions(f"xgboost_{macro}", stages=["Production"])[0]
            model_s3_uri = model_version.source

            # Extract S3 bucket and key
            s3_bucket, s3_key = model_s3_uri.replace("s3://", "").split("/", 1)
            local_model_path = self.model_cache_dir / f"xgboost_{macro}.tar.gz"
            extracted_model_path = self.model_cache_dir / f"xgboost_{macro}"

            # Download model if not already cached
            if not extracted_model_path.exists():
                logger.info(f"Downloading {macro} model from {model_s3_uri}...")
                self.s3_client.download_file(s3_bucket, s3_key, str(local_model_path))

                # Extract model
                os.system(f"tar -xzf {local_model_path} -C {self.model_cache_dir}")

            # Load model
            try:
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(str(extracted_model_path / "xgboost-model"))
                self.models[macro] = model
                logger.info(f"Loaded latest production model for {macro}")
            except Exception as e:
                logger.error(f"Error loading model for {macro}: {e}")
                raise
        
    def _load_local_transformers(self):
        try:
            self.tfidf = load(self.tfidf_path)
            self.svd = load(self.svd_path)
            logger.info("Local transformer objects loaded successfully")
        except Exception as e:
            logger.error(f"Error loading local transformer objects: {e}")
            raise
        
    def _load_production_transformers(self):
        """Download transformers from S3"""
        try:
            transformers = ["tfidf_transformer.joblib", "svd_transformer.joblib"]
            s3_bucket = "iifymate-ml-data"
            
            for transformer in transformers:
                local_path = self.transformer_dir / transformer
                s3_path = f"feature_Store/transformers/{transformer}"
                
                # Download only if not already cached
                if not local_path.exists():
                    logger.info(f"Downloading {transformer} from S3...")
                    self.s3_client.download_file(s3_bucket, s3_path, str(local_path))
                    
            # Load models
            self.tfidf = joblib.load(self.transformer_dir / "tfidf_transformer.joblib")
            self.svd = joblib.load(self.transformer_dir / "svd_transformer.joblib")
            logger.info("Production transformer objects loaded successfully")
        except Exception as e:
            logger.error(f"Error loading production transformer objects: {e}")
            raise
            

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
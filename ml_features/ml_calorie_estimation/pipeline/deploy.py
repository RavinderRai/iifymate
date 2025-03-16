import os
import json
import mlflow
import requests
import logging
from ml_features.ml_calorie_estimation.src.databases.config import DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GitHub Actions API Setup
GITHUB_REPO = "RavinderRai/iifymate"
GITHUB_WORKFLOW = "deploy_model.yml"  # Matches your GitHub Actions workflow file
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/dispatches"

# GitHub Token stored in a local environment variable
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")  # Make sure you set this in your environment

database_config = DatabaseConfig(
    username=os.getenv("RDS_USERNAME"),
    password=os.getenv("RDS_PASSWORD"),
    host=os.getenv("RDS_HOST"),
    database="mlflowdb"
)
tracking_uri = f"postgresql://{database_config.username}:{database_config.password}@{database_config.host}/{database_config.database}"
mlflow.set_tracking_uri(tracking_uri)

MODEL_NAMES = ['Fat', 'Carbohydrates_net', 'Protein']

def fetch_latest_model_versions():
    """
    Retrieves the latest S3 model paths for each macronutrient from MLflow.
    Ensures all paths are valid S3 locations.
    
    Returns:
        List of latest model S3 paths.
    """
    client = mlflow.MlflowClient()
    models_info = {}

    for macro in MODEL_NAMES:
        model_name = f"xgboost_target_{macro}"
        
        try:
            registered_model = client.get_registered_model(model_name)
            latest_version = max([int(v.version) for v in registered_model.latest_versions])

            model_details = client.get_model_version(model_name, str(latest_version))
            s3_path = model_details.source
            
            if not s3_path.startswith("s3://"):
                raise ValueError(f"Model source is not an S3 path: {s3_path}")

            logger.info(f"✅ Latest model for {macro}: v{latest_version} - {s3_path}")
            models_info[model_name] = {"s3_path": s3_path, "version": latest_version}

        except Exception as e:
            logger.error(f"❌ Error fetching model {model_name}: {e}")

    if len(models_info) != len(MODEL_NAMES):
        logger.error("❌ Not all models were found. Aborting deployment.")
        return None

    return models_info
    
def trigger_github_actions(model_data: dict):
    """
    Sends a request to GitHub Actions to start deployment.
    
    Args:
        model_data (dict): Dictionary with model names, S3 paths, and versions.
    """
    headers = {
        "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "ref": "main",
        "inputs": {"model_data": json.dumps(model_data)}
    }

    try:
        response = requests.post(GITHUB_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        logger.info("✅ GitHub Actions deployment triggered successfully!")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Deployment failed: {e}")
        logger.error(f"🔍 GitHub Response: {response.text}")

if __name__ == "__main__":
    model_s3_paths = fetch_latest_model_versions()
    if model_s3_paths:
        logger.info("✅ All models found. Triggering deployment...")
        trigger_github_actions(model_s3_paths)
    else:
        logger.error("❌ Deployment aborted due to missing models.")

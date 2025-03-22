import os
import subprocess
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Setup logging to console only (no file storage)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()

class ModelUpdateRequest(BaseModel):
    model_data: dict[str, dict[str, str]]  # Expecting a dictionary of models with S3 paths & versions


@app.post("/update_model")
async def update_model(data: ModelUpdateRequest):
    """
    Receives a model update request, logs model metadata, 
    and restarts the `ml_calorie_predictor` service.
    """
    try:
        # Log the received model data
        logging.info("🚀 Received model update request:")
        for model_name, details in data.model_data.items():
            logging.info(f"🔹 Model: {model_name} | Version: {details['version']} | S3 Path: {details['s3_path']}")


        # Navigate to the project directory
        os.chdir("/home/ubuntu/iifymate")

        # Rebuild the `ml_calorie_predictor` service
        logging.info("🔨 Rebuilding the ml_calorie_predictor service...")
        rebuild_result = subprocess.run(["docker-compose", "build", "ml_calorie_predictor"], capture_output=True, text=True)

        if rebuild_result.returncode == 0:
            logging.info(f"✅ Docker Build Successful: {rebuild_result.stdout.strip()}")
        else:
            logging.error(f"❌ Docker Build Failed: {rebuild_result.stderr.strip()}")
            raise HTTPException(status_code=500, detail={"status": "Docker Build Failed", "error": rebuild_result.stderr.strip()})

        # Restart the `ml_calorie_predictor` service (without affecting others)
        logging.info("♻️ Restarting ml_calorie_predictor service...")
        restart_result = subprocess.run(["docker-compose", "up", "-d", "--no-deps", "ml_calorie_predictor"], capture_output=True, text=True)

        if restart_result.returncode == 0:
            logging.info(f"✅ Docker Restart Successful: {restart_result.stdout.strip()}")
        else:
            logging.error(f"❌ Docker Restart Failed: {restart_result.stderr.strip()}")
            raise HTTPException(status_code=500, detail={"status": "Docker Restart Failed", "error": restart_result.stderr.strip()})

        return {"status": "✅ Model updated and service restarted"}

    except Exception as e:
        logging.exception(f"🔥 Exception occurred: {str(e)}")
        return {"status": "❌ Internal Server Error", "error": str(e)}

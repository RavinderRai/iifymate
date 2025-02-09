import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from pathlib import Path
from ml_features.ml_calorie_estimation.src.predicting.macro_predictor import MacroPredictor
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

ML_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MLFLOW_TRACKING_URI = os.path.join(ML_PROJECT_ROOT, "mlruns")

app = FastAPI()
predictor = MacroPredictor(MLFLOW_TRACKING_URI)

REQUESTS = Counter('calorie_predictions_total', 'Total number of predictions made')
PREDICTION_TIME = Histogram('prediction_latency_seconds', 'Time spent processing prediction')

class TextInput(BaseModel):
    text: str
    
class BatchTextInput(BaseModel):
    texts: list[str]
    
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
@app.post("/predict", response_model=dict[str, float])
async def predict(input_data: TextInput) -> JSONResponse:
    try:
        # Increment requests counter
        REQUESTS.inc()
        with PREDICTION_TIME.time():
            result = predictor.predict(input_data.text)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/batch-predict", response_model=list[dict[str, float]])
async def batch_predict(input_data: BatchTextInput) -> JSONResponse:
    try:
        result = predictor.batch_predict(input_data.texts)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # uvicorn ml_features.ml_calorie_estimation.pipeline.predict:app --host 0.0.0.0 --port 8000 --reload
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from tempfile import NamedTemporaryFile
import logging
from ml_features.llm_calorie_estimation.src.services.meal_analyzer import MealAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Calorie Estimation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    app.state.meal_analyzer = MealAnalyzer(api_key=api_key)
    logger.info("MealAnalyzer initialized successfully")
    
@app.post("/analyze-meal/")
async def analyze_meal(file: UploadFile = File(...)) -> dict:
    """Endpoint to analyze a meal image and return the combined features string for ML model input"""
    try:
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            
        try:
            result = app.state.meal_analyzer.get_ml_features(temp_path)
            
            return {
                "status": "success",
                "data": {
                    "combined_features": result
                }
            }
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error analyzing meal: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # To test this, run this in WSL in root directory:
    # uvicorn ml_features.llm_calorie_estimation.src.api.app:app --reload --port 8001
    # Then, try this sample input in another terminal:
    # curl -X POST "http://localhost:8000/analyze-meal/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@notebooks/data/sample_meal_images/scrambled_eggs.jpg"
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
    # Or change the port for docker
    # curl -X POST "http://localhost:8001/analyze-meal/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@notebooks/data/sample_meal_images/scrambled_eggs.jpg"
    
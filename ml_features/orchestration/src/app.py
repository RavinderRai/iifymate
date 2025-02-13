from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

app = FastAPI(title="ML Features Orchestration Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CalorieEstimationOrchestrator:
    def __init__(self):
        # For local testing
        #self.ml_service_url = "http://localhost:8000" 
        #self.llm_service_url = "http://localhost:8001" #"http://llm_calorie_estimation:8001"
        
        #For Docker
        self.ml_service_url = "http://ml_calorie_predictor:8000"
        self.llm_service_url = "http://llm_calorie_estimation:8001"
        self.client = httpx.AsyncClient(follow_redirects=True)
        
    async def process_image(self, file: UploadFile) -> Dict:
        try:
            logger.info("Sending image to LLM service for analysis")
            files = {"file": (file.filename, file.file, file.content_type)}
            
            # Step 1: Send image to LLM service
            try:
                llm_response = await self.client.post(
                    f"{self.llm_service_url}/analyze-meal/",
                    files=files
                )
                llm_response.raise_for_status()
                llm_data = llm_response.json()
                logger.info(f"LLM Response received: {llm_data}")  # Add this log
                
                if not isinstance(llm_data, dict):
                    raise ValueError(f"Expected dict response from LLM, got {type(llm_data)}")
                if "data" not in llm_data:
                    raise ValueError(f"No 'data' key in LLM response: {llm_data}")
                if "combined_features" not in llm_data.get("data", {}):
                    raise ValueError(f"No 'combined_features' in LLM response data: {llm_data}")
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error in LLM service call: {str(e)}")
                if hasattr(e, 'response'):
                    logger.error(f"LLM Response Status: {e.response.status_code}")
                    logger.error(f"LLM Response Content: {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}", exc_info=True)
                raise
            
            # Step 2: Send features to ML service
            try:
                ml_request_data = {"text": llm_data["data"]["combined_features"]}
                logger.info(f"Sending to ML service: {ml_request_data}")
                
                ml_response = await self.client.post(
                    f"{self.ml_service_url}/predict",
                    json=ml_request_data,
                    timeout=30.0  # Add explicit timeout
                )
                logger.info(f"ML service status code: {ml_response.status_code}")
                
                ml_response.raise_for_status()
                ml_data = ml_response.json()
                logger.info(f"ML Response received: {ml_data}")
                
                if not ml_data:
                    raise ValueError("Empty response from ML service")
                    
                return ml_data

            except httpx.HTTPError as e:
                logger.error(f"HTTP error in ML service call: {str(e)}")
                if hasattr(e, 'response'):
                    logger.error(f"ML Response Status: {e.response.status_code}")
                    logger.error(f"ML Response Content: {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error in ML service call: {type(e).__name__}: {str(e)}", exc_info=True)
                raise
            
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            raise HTTPException(status_code=500, detail="Error communicating with services")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

orchestrator = CalorieEstimationOrchestrator()

@app.post("/estimate-calories/")
async def estimate_calories(file: UploadFile = File(...)) -> dict:
    """Upload a meal and get calorie estimation"""
    return await orchestrator.process_image(file)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # To test this run:
    # uvicorn ml_features.orchestration.src.app:app --reload --port 8002
    # Then, try this sample input in another terminal:
    # curl -X POST "http://localhost:8002/estimate-calories/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@notebooks/data/sample_meal_images/scrambled_eggs.jpg"
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
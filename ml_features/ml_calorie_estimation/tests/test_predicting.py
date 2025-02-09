import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import mlflow

# Import the app after mocking
with patch('ml_features.ml_calorie_estimation.src.predicting.macro_predictor.MacroPredictor') as MockPredictor:
    # Configure the mock
    predictor_mock = Mock()
    predictor_mock.predict.return_value = {
        'fat': 10,
        'protein': 20,
        'carbs': 30,
        'calories': 370
    }
    predictor_mock.batch_predict.return_value = [
        {
            'fat': 10,
            'protein': 20,
            'carbs': 30,
            'calories': 370
        }
    ]
    MockPredictor.return_value = predictor_mock
    
    # Now import the app
    from ml_features.ml_calorie_estimation.pipeline.predict import app

# Test data
SAMPLE_TEXT = "Vegan Black Bean Tacos with rice"
SAMPLE_PREDICTION = {
    'fat': 10,
    'protein': 20,
    'carbs': 30,
    'calories': 370
}

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)

def test_predict_endpoint(client):
    """Test the /predict endpoint"""
    response = client.post("/predict", json={"text": SAMPLE_TEXT})
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.content}")
    assert response.status_code == 200
    assert response.json() == SAMPLE_PREDICTION

def test_batch_predict_endpoint(client):
    """Test the /batch-predict endpoint"""
    response = client.post("/batch-predict", json={"texts": [SAMPLE_TEXT]})
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.content}")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0] == SAMPLE_PREDICTION

def test_metrics_endpoint(client):
    """Test the /metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200

if __name__ == '__main__':
    # Run with pytest with this command:
    # pytest ml_features/ml_calorie_estimation/tests/test_predicting.py
    pytest.main([__file__, '-v'])

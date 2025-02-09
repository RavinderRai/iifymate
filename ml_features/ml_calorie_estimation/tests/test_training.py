import pytest
import pandas as pd
import numpy as np
import warnings
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
import mlflow
import xgboost as xgb

from ml_features.ml_calorie_estimation.src.training.data_validation import clean_training_testing_data
from ml_features.ml_calorie_estimation.src.training.grid_search import grid_search_macro_model
from ml_features.ml_calorie_estimation.src.training.model_utils import train_macro_model, evaluate_model
from ml_features.ml_calorie_estimation.src.training.multi_train import train_all_macro_models

# Filter out specific deprecation warnings from dependencies
warnings.filterwarnings('ignore', category=DeprecationWarning, module='google._upb._message')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Constants
SAMPLE_SIZE = 100
N_FEATURES = 50
MACROS = ['target_Fat', 'target_Carbohydrates_net', 'target_Protein']

@pytest.fixture(scope="session")
def temp_mlruns_dir():
    """Create a temporary directory for MLflow runs and set up test experiment"""
    temp_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file://{temp_dir}")
    
    # Create test experiment
    test_experiment_name = "test_experiment"
    mlflow.create_experiment(test_experiment_name)
    mlflow.set_experiment(test_experiment_name)
    
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_data():
    """Create synthetic data for testing"""
    np.random.seed(42)
    
    # Create feature matrix with component_ prefix
    X = pd.DataFrame(
        np.random.randn(SAMPLE_SIZE, N_FEATURES),
        columns=[f'component_{i}' for i in range(N_FEATURES)]
    )
    
    # Create target dataframe with realistic macro values
    y = pd.DataFrame({
        'target_Fat': np.random.uniform(0, 100, SAMPLE_SIZE),
        'target_Carbohydrates_net': np.random.uniform(0, 200, SAMPLE_SIZE),
        'target_Protein': np.random.uniform(0, 150, SAMPLE_SIZE)
    })
    
    return X, y

@pytest.fixture
def mock_mlflow_run():
    """Create a mock MLflow run context"""
    mock_run = MagicMock()
    mock_run.__enter__.return_value = Mock()
    mock_run.__exit__.return_value = None
    return mock_run

@pytest.fixture
def corrupt_data(sample_data):
    """Create data with NaN and inf values"""
    X, y = sample_data
    X_corrupt = X.copy()
    y_corrupt = y.copy()
    
    # Insert some NaN and inf values
    X_corrupt.iloc[0, 0] = np.nan
    X_corrupt.iloc[1, 1] = np.inf
    y_corrupt.iloc[2, 0] = np.nan
    
    return X_corrupt, y_corrupt

def test_clean_training_testing_data(corrupt_data):
    """Test data cleaning functionality"""
    X_corrupt, y_corrupt = corrupt_data
    X_clean, y_clean = clean_training_testing_data(X_corrupt, y_corrupt, MACROS)
    
    # Check that NaN and inf values were removed
    assert not X_clean.isna().any().any()
    assert not np.isinf(X_clean).any().any()
    assert not y_clean.isna().any().any()
    
    # Check that rows were removed
    assert len(X_clean) < len(X_corrupt)
    assert len(X_clean) == len(y_clean)

@patch('mlflow.start_run')
@patch('mlflow.log_params')
@patch('mlflow.log_metric')
def test_grid_search_macro_model(mock_log_metric, mock_log_params, mock_start_run, 
                                sample_data, mock_mlflow_run, temp_mlruns_dir):
    """Test grid search functionality"""
    X, y = sample_data
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01]
    }
    
    mock_start_run.return_value = mock_mlflow_run
    
    best_params = grid_search_macro_model(
        X, y, 'target_Fat', param_grid, is_dev=True
    )
    
    # Verify MLflow interactions
    assert mock_log_params.called
    assert mock_log_metric.called
    
    # Check returned parameters
    assert isinstance(best_params, dict)
    assert 'max_depth' in best_params
    assert 'learning_rate' in best_params

@patch('mlflow.start_run')
@patch('mlflow.xgboost.log_model')
def test_train_macro_model(mock_log_model, mock_start_run, sample_data, 
                          mock_mlflow_run, temp_mlruns_dir):
    """Test model training functionality"""
    X, y = sample_data
    model_params = {'max_depth': 3, 'learning_rate': 0.01}
    
    mock_start_run.return_value = mock_mlflow_run
    
    model = train_macro_model(
        X, y, 'target_Fat', 'test_model', model_params
    )
    
    # Verify model type and MLflow interactions
    assert isinstance(model, xgb.XGBRegressor)
    assert mock_log_model.called

@patch('mlflow.log_metric')
def test_evaluate_model(mock_log_metric, sample_data, temp_mlruns_dir):
    """Test model evaluation functionality"""
    X, y = sample_data
    
    # Train a simple model for testing
    model = xgb.XGBRegressor(max_depth=3)
    model.fit(X, y['target_Fat'])
    
    metrics = evaluate_model(model, X, y, 'target_Fat')
    
    # Check metrics structure
    assert 'r2' in metrics
    assert 'mse' in metrics
    assert isinstance(metrics['r2'], float)
    assert isinstance(metrics['mse'], float)
    assert mock_log_metric.called

class MockRunContext:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

@patch('mlflow.set_experiment')
def test_train_all_macro_models(mock_set_experiment, sample_data, temp_mlruns_dir):
    """Test end-to-end training pipeline"""
    X, y = sample_data
    X_test, y_test = sample_data  # Using same data for simplicity
    
    # Create a more sophisticated mock for MLflow runs
    with patch('mlflow.start_run') as mock_start_run:
        # Each call to start_run will return a new mock context
        mock_start_run.side_effect = [MockRunContext() for _ in range(12)]  # Adjust number based on expected runs
        
        # Mock other MLflow functions
        with patch('mlflow.log_params'), \
             patch('mlflow.log_metrics'), \
             patch('mlflow.log_param'), \
             patch('mlflow.log_metric'), \
             patch('mlflow.xgboost.log_model'), \
             patch('mlflow.set_tag'):
            
            models, metrics = train_all_macro_models(
                X, X_test, y, y_test, is_dev=True
            )
    
    # Check results structure
    assert len(models) == len(MACROS)
    assert len(metrics) == len(MACROS)
    
    for macro in MACROS:
        assert macro in models
        assert macro in metrics
        assert 'r2' in metrics[macro]
        assert 'mse' in metrics[macro]
        assert 'best_params' in metrics[macro]

if __name__ == '__main__':
    # Run with pytest with this command:
    # pytest ml_features/ml_calorie_estimation/tests/test_training.py
    pytest.main([__file__, '-v'])
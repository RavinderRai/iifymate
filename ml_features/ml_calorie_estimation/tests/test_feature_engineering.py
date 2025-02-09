import pytest
from pathlib import Path
import pandas as pd
import ast
import json

from ml_features.ml_calorie_estimation.src.feature_engineering.data_transformations import comma_to_bracket, replace_with_priority, get_macros


@pytest.fixture
def sample_recipes():
    """Load 100 random recipes from clean data"""
    sample_data_path = Path(__file__).parent / 'data' / 'test_clean_data.csv'
    df = pd.read_csv(sample_data_path)
    return df

def test_comma_to_bracket(sample_recipes):
    ingredientLines = sample_recipes['ingredientLines']
    
    # Note since we saved a sample dataset in csv format, our columns of lists are saved as str objects
    # We need to convert them back to lists to test the function
    ingredientLines = ingredientLines.apply(ast.literal_eval)
    
    ingredientLines = ingredientLines.apply(comma_to_bracket)
    
    assert isinstance(ingredientLines, pd.Series)
    assert ingredientLines.dtype == 'object'
    assert all(isinstance(x, str) for x in ingredientLines)
    
def test_replace_with_priority(sample_recipes):
    healthLabels = sample_recipes['healthLabels']
    
    # Note since we saved a sample dataset in csv format, our columns of lists are saved as str objects
    # We need to convert them back to lists to test the function
    healthLabels = healthLabels.apply(ast.literal_eval)
    
    healthLabels = healthLabels.apply(replace_with_priority)
    
    assert isinstance(healthLabels, pd.Series)
    assert healthLabels.dtype == 'object'
    assert all(isinstance(x, str) for x in healthLabels)
    
def test_get_macros(sample_recipes):
    nutrients = sample_recipes['totalNutrients']
        
    # Note since we saved a sample dataset in csv format, our columns of lists are saved as str objects
    # We need to convert them back to lists to test the function
    nutrients = nutrients.apply(json.loads)
    
    assert all(isinstance(x, dict) for x in nutrients)
    
    nutrients = list(nutrients.apply(lambda row: get_macros(row)))
    
    assert all(isinstance(x, dict) for x in nutrients)
    
if __name__ == '__main__':
    # Run with pytest with this command:
    # pytest ml_features/ml_calorie_estimation/tests/test_feature_engineering.py
    pytest.main([__file__])
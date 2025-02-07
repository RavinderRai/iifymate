import pytest
import pandas as pd
import ast
import json
from pathlib import Path

from ml_features.ml_calorie_estimation.src.data_cleaning.recipe_cleaner import (
    drop_unused_columns,
    rename_columns,
    clean_ingredients,
    convert_numeric_columns,
    clean_nutrients,
    remove_faulty_nutrients
)

@pytest.fixture
def sample_recipes():
    """Load first two recipes from sample data"""
    sample_data_path = Path(__file__).parent / 'data' / 'test_raw_data.csv'
    df = pd.read_csv(sample_data_path)
    return df

def test_drop_unused_columns(sample_recipes):
    result = drop_unused_columns(sample_recipes)
    
    assert 'uri' not in result.columns
    assert 'url' not in result.columns
    assert 'cautions' not in result.columns
    assert 'TotalDaily' not in result.columns
    assert 'digest' not in result.columns
    
def test_rename_columns(sample_recipes):
    result = rename_columns(sample_recipes)
    
    assert 'serving_size' in result.columns
    
def test_clean_ingredients(sample_recipes):
    result = clean_ingredients(sample_recipes)
    
    assert isinstance(result['ingredients'][0], list)
    
def test_remove_faulty_nutrients(sample_recipes):
    """Test removing records with invalid nutrient data using real sample data"""
    # First clean the nutrients data so it's in the right format
    sample_recipes['totalNutrients'] = sample_recipes['totalNutrients'].apply(ast.literal_eval)
    
    # Apply the function
    result = remove_faulty_nutrients(sample_recipes)
    
    # Test that we still have data after cleaning
    assert len(result) > 0
    
    # Verify the structure and validity of remaining data
    for _, row in result.iterrows():
        nutrients = row['totalNutrients']
        
        # Check each required nutrient
        for nutrient in ['FAT', 'CHOCDF.net', 'PROCNT']:
            # Nutrient exists
            assert nutrient in nutrients
            
            # Has quantity and it's not negative
            assert 'quantity' in nutrients[nutrient]
            assert nutrients[nutrient]['quantity'] >= 0
            
            # Has correct unit
            assert 'unit' in nutrients[nutrient]
            assert nutrients[nutrient]['unit'] == 'g'
    
def test_full_pipeline_with_real_data(sample_recipes):
    """Test the entire cleaning pipeline with real data"""
    df = sample_recipes.copy()
    
    df['totalNutrients'] = df['totalNutrients'].apply(json.loads)
    
    # Apply all transformations
    df = remove_faulty_nutrients(df)
    df = drop_unused_columns(df)
    df = rename_columns(df)
    df = clean_ingredients(df)
    df = convert_numeric_columns(df)
    df = clean_nutrients(df)
    
    # Basic sanity checks
    assert 'uri' not in df.columns
    assert 'serving_size' in df.columns
    assert df['calories'].dtype == 'float64'
    assert isinstance(df['ingredients'].iloc[0], list)
    assert isinstance(df['totalNutrients'].iloc[0], dict)
    
if __name__ == '__main__':
    pytest.main([__file__])
    
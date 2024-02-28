import pandas as pd
import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from data_processing import get_target_variable, preprocess_dish_type, preprocess_meal_type, get_training_testing_data

@pytest.fixture
def test_data():
    return pd.read_csv('sample_recipes.csv')

def test_get_target_variable_dataframe_size(test_data):
    binned_calories_df = get_target_variable(test_data)
    assert binned_calories_df.shape == (365, 23)
    return binned_calories_df

def test_preprocess_dish_type_output_shape(test_data, test_get_target_variable_dataframe_size):
    dish_df, _, _ = preprocess_data(raw_data)
    assert dish_df.shape == (359, 25)

def test_preprocess_meal_type_output_shape(test_data, test_preprocess_dish_type_output_shape):
    meal_df = preprocess_meal_type(raw_data)
    assert meal_df.shape == (359, 26)
    return meal_df


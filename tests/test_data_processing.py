import pandas as pd
import pytest

import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


@pytest.fixture
def test_data():
    return pd.read_csv('sample_recipes.csv')

sys.path.append("../calorie_predicter/")

from calorie_predicter.data_processing import get_target_variable, preprocess_dish_type, preprocess_meal_type, get_training_testing_data, pre_process_text
from calorie_predicter.utils import one_hot_encode
from calorie_predicter import utils

@pytest.fixture
def binned_calories_data(test_data):
    return get_target_variable(test_data)

@pytest.fixture
def pre_processed_dish_data(binned_calories_data):
    dish_df, _, _ = preprocess_dish_type(binned_calories_data)
    return dish_df

@pytest.fixture
def pre_processed_meal_data(pre_processed_dish_data):
    meal_df = preprocess_meal_type(pre_processed_dish_data)
    return meal_df

@pytest.fixture
def onehot_encoded_data(pre_processed_meal_data):
    onehot_encoded_df, _ = one_hot_encode(pre_processed_meal_data, 'mealTypeRefined')
    onehot_encoded_df = pd.concat([pre_processed_meal_data, onehot_encoded_df], axis=1)
    return onehot_encoded_df

@pytest.fixture
def pre_processed_text_data(onehot_encoded_data):
    english_stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    tokenized_label_df = pre_process_text(
        df=onehot_encoded_data,
        column='label',
        stop_words=english_stop_words,
        lemmatizer=lemmatizer,
        tokenizer=word_tokenize
    )
    return tokenized_label_df

@pytest.fixture
def training_testing_splits_data(pre_processed_text_data):
    pre_processed_text_data['dishTypeSkewedLabels'] = pre_processed_text_data['dishTypeSkewedLabels'].astype(int)
    pre_processed_text_data['calorieLabels'] = pre_processed_text_data['binnedCalories'].astype(int)
    
    X_cols = ['mealTypeRefined_breakfast', 
              'mealTypeRefined_lunch/dinner', 
              'mealTypeRefined_snack', 'label', 
              'dishTypeSkewedLabels']
    y_col = 'binnedCalories'
    X_train, X_test, y_train, y_test, _ = get_training_testing_data(pre_processed_text_data, X_cols, y_col)
    return X_train, X_test, y_train, y_test

    

def test_sample_data_shape(test_data):
    assert test_data.shape == (401, 22)

def test_get_target_variable_dataframe_size(binned_calories_data):
    assert binned_calories_data.shape == (365, 23)

def test_preprocess_dish_type_output_shape(pre_processed_dish_data):
    assert pre_processed_dish_data.shape == (359, 25)

def test_preprocess_meal_type_output_shape(pre_processed_meal_data):
    assert pre_processed_meal_data.shape == (359, 26)

def test_one_hot_encoder_output_shape(onehot_encoded_data):
    assert onehot_encoded_data.shape == (359, 29)

def test_pre_processed_text_data_output_shape(pre_processed_text_data):
    assert pre_processed_text_data.shape == (359, 29)

def test_training_testing_splits_data_output_shape(training_testing_splits_data):
    X_train, X_test, y_train, y_test = training_testing_splits_data
    assert X_train.shape == (287, 463)
    assert X_test.shape == (72, 463)
    assert y_train.shape == (287,)
    assert y_test.shape == (72,)
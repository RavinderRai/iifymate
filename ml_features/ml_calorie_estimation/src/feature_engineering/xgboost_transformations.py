import numpy as np
import pandas as pd
import logging
from nltk.tokenize import word_tokenize
from ml_features.ml_calorie_estimation.src.feature_engineering.text_processing import remove_stop_words, lemmatizing, get_tfidf_splits, SVD_reduction
from ml_features.ml_calorie_estimation.src.feature_engineering.data_transformations import comma_to_bracket, replace_with_priority, get_macros
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def xgboost_transformations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, any, any]:
    """
    Applies feature engineering transformations on the input DataFrame for training an XGBoost model.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the 'ingredientLines', 'healthLabels', 'label', and 'totalNutrients' columns.
    
    Returns:
    X_train (pd.DataFrame): The Pandas DataFrame containing the input features for training.
    X_test (pd.DataFrame): The Pandas DataFrame containing the input features for testing.
    y_train (pd.DataFrame): The Pandas DataFrame containing the target variables for training.
    y_test (pd.DataFrame): The Pandas DataFrame containing the target variables for testing.
    tfidf_fitted (any): The fitted TF-IDF vectorizer model.
    svd_fitted (any): The fitted SVD reduction model.
    """
    logger.info("Starting feature transformation process for an XGBoost model.")
    
    # Get relevant features
    logger.info("Extracting relevant features.")
    ingredientLines = df['ingredientLines']
    healthLabels = df['healthLabels']
    nutrients = df['totalNutrients']
    
    # Feature engineering transformation code here
    logger.info("Applying feature engineering transformations on ingredient lines and health labels.")
    ingredientLines = ingredientLines.apply(comma_to_bracket)
    healthLabels = healthLabels.apply(replace_with_priority)
    
    # Get X and y data    
    logger.info("Concatenating features to form input data.")
    X = healthLabels + " " + df['label'] + " " + ingredientLines
    X = X.rename('fullRecipeInput')
    
    logger.info("Applying text processing transformations.")
    X = X.apply(remove_stop_words)
    X = X.apply(lemmatizing)
    X = X.apply(lambda x: word_tokenize(x))
    
    logger.info("Extracting target variables.")
    y = pd.DataFrame(list(nutrients.apply(lambda row: get_macros(row))))
    y.rename(columns={'Carbohydrates (net)': 'Carbohydrates_net'}, inplace=True)
    
    # Split data into training and testing sets and perform TF-IDF vectorization
    logger.info("Performing train-test split and TF-IDF vectorization.")
    X_train, X_test, y_train, y_test, tfidf_fitted = get_tfidf_splits(X, y)

    logger.info("Applying SVD reduction.")
    X_train, X_test, svd_fitted = SVD_reduction(X_train, X_test, n_components=500)

    logger.info("Applying log transformation to target variables.")
    y_train, y_test = np.log1p(y_train), np.log1p(y_test)

    return X_train, X_test, y_train, y_test, tfidf_fitted, svd_fitted
import numpy as np
import pandas as pd
import logging
from nltk.tokenize import word_tokenize
from joblib import dump

from ml_features.ml_calorie_estimation.src.data_ingestion.utils import create_db_config, load_config
from ml_features.ml_calorie_estimation.src.databases.manager import DatabaseManager
from ml_features.ml_calorie_estimation.src.feature_engineering.text_processing import remove_stop_words, lemmatizing, get_tfidf_splits, SVD_reduction
from ml_features.ml_calorie_estimation.src.feature_engineering.data_transformations import comma_to_bracket, replace_with_priority, get_macros
from ml_features.ml_calorie_estimation.src.databases.models.clean_data import CleanRecipe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: need to implement local and production environments specifically for feature engineering
def run_feature_transformations(env: str = "local"):
    logger.info("Starting feature transformation process.")
    
    # Load data from database
    logger.info("Loading clean recipe data from database.")
    config = load_config(env)
    db_config = create_db_config(config.database, env=env)
    db_manager = DatabaseManager(db_config)
    session = db_manager.Session()
    
    query = session.query(CleanRecipe).statement
    df = pd.read_sql(query, session.bind)
    logger.info("Successfully loaded clean recipe data.")

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
    y.rename(columns={'Carbohydrates (net)': 'Carbohydrates_net'},inplace=True)
    
    # Split data into training and testing sets and perform TF-IDF vectorization
    logger.info("Performing train-test split and TF-IDF vectorization.")
    X_train, X_test, y_train, y_test, tfidf_fitted = get_tfidf_splits(X, y)

    logger.info("Applying SVD reduction.")
    X_train, X_test, svd_fitted = SVD_reduction(X_train, X_test, n_components=500)

    logger.info("Applying log transformation to target variables.")
    y_train, y_test = np.log1p(y_train), np.log1p(y_test)
    
    # Saving to feature store
    logger.info("Preparing data for saving to feature store.")
    feature_store_path = "ml_features/ml_calorie_estimation/feature_store/feature_repo"
    data_path = f"{feature_store_path}/data"
    
    feature_df = X_train.copy()
    feature_df['recipe_id'] = feature_df.index
    feature_df['timestamp'] = pd.Timestamp.now()
    
    # Add target variables
    for col in y_train.columns:
        feature_df[f'target_{col}'] = y_train[col]
    
    logger.info("Saving features to parquet file for Feast.")
    feature_path = f"{data_path}/recipe_features.parquet"
    feature_df.to_parquet(feature_path)
    
    # Also save test data separately
    test_df = X_test.copy()
    test_df['recipe_id'] = test_df.index
    test_df['timestamp'] = pd.Timestamp.now()
    
    for col in y_test.columns:
        test_df[f'target_{col}'] = y_test[col]
    test_df.to_parquet(f"{data_path}/test_features.parquet")
    
    # Save fitted transformers separately
    logger.info("Saving fitted transformers.")
    dump(tfidf_fitted, f"{data_path}/tfidf_fitted.joblib")
    dump(svd_fitted, f"{data_path}/svd_fitted.joblib")
    
    logger.info(f"Completed feature transformation process. Saved features to {feature_path}")
    
    
if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.pipeline.feature_engineering
    import os
    environment = os.getenv("ENV", "local")
    run_feature_transformations(env=environment)
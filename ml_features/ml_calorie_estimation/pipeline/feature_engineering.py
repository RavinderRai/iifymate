import os
import pandas as pd
import logging

from ml_features.ml_calorie_estimation.src.data_ingestion.utils import create_db_config, load_config
from ml_features.ml_calorie_estimation.src.databases.manager import DatabaseManager
from ml_features.ml_calorie_estimation.src.feature_engineering.xgboost_transformations import xgboost_transformations
from ml_features.ml_calorie_estimation.src.databases.models.clean_data import CleanRecipe
from ml_features.ml_calorie_estimation.src.feature_engineering.feature_store import MLFeatureStore

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
    logger.info("Successfully loaded CleanRecipe data.")
    
    # Perform feature transformations
    logger.info("Performing feature transformations for XGBoost model.")
    X_train, X_test, y_train, y_test, tfidf_fitted, svd_fitted = xgboost_transformations(df)
        
    feature_df = X_train.copy()
    feature_df['recipe_id'] = feature_df.index
    feature_df['timestamp'] = pd.Timestamp.now()
        
    # Add target variables
    for col in y_train.columns:
        feature_df[f'target_{col}'] = y_train[col]
    logger.info("Feature transformation of X data completed with shape: {}".format(feature_df.shape))
        
    # Also save test data separately
    test_df = X_test.copy()
    test_df['recipe_id'] = test_df.index
    test_df['timestamp'] = pd.Timestamp.now()
    
    for col in y_test.columns:
        test_df[f'target_{col}'] = y_test[col]
    logger.info("Feature transformation of Y data completed with shape: {}".format(test_df.shape))
    
    logger.info("Initializing feature store.")
    feature_store = MLFeatureStore(env)
    
    logger.info("Saving features and transformers to feature store.")    
    feature_store.save_transformer(tfidf_fitted, "tfidf_transformer.joblib")
    feature_store.save_transformer(svd_fitted, "svd_transformer.joblib")
    feature_store.save_features(feature_df, "recipe_features.parquet")
    feature_store.save_features(test_df, "test_features.parquet")
    
    
if __name__ == "__main__":
    # Run this command in WSL in root directory to test:
    # python -m ml_features.ml_calorie_estimation.pipeline.feature_engineering
    import os
    environment = os.getenv("ENV", "local")
    run_feature_transformations(env=environment)
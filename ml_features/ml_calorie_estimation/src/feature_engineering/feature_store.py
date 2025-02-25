import os
from pathlib import Path
import tempfile
import pandas as pd
import logging
from joblib import dump
import boto3
from botocore.exceptions import ClientError

from ml_features.ml_calorie_estimation.src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFeatureStore:
    def __init__(self, env: str = "local"):
        self.env = env
        self.config = load_config(env)
        
        if env == "production":
            if not self.config.aws:
                raise ValueError("AWS configuration is missing")
            self.s3_client = boto3.client('s3')
            self.bucket_name = self.config.aws.s3_bucket
            self.feature_store_prefix = self.config.aws.feature_store_prefix
            
        elif env == "local":
            self.base_path = Path("ml_features/ml_calorie_estimation/feature_store/feature_repo/data")
            self.base_path.mkdir(parents=True, exist_ok=True)
            
        else:
            self.base_path = Path("/tmp/feature_store") # Temporary location
            self.base_path.mkdir(parents=True, exist_ok=True)
        
    def _save_features_locally(self, feature_df: pd.DataFrame, filename: str):
        """Save features locally"""
        local_path = self.base_path / filename
        feature_df.to_parquet(local_path)
        logger.info(f"Saved features to {local_path}")
        
    def _save_transformers_locally(self, transformer, filename: str):
        """Save transformers locally"""
        local_path = self.base_path / filename
        dump(transformer, local_path)
        logger.info(f"Saved transformers to {local_path}")
                
    def _save_features_to_s3(self, feature_df: pd.DataFrame, filename: str):
        """Save features to S3"""
        try:
            s3_path = f"s3://{self.bucket_name}/{self.feature_store_prefix}{filename}"
            feature_df.to_csv(s3_path, index=False)
            logger.info(f"Saved features to {s3_path}")
        except Exception as e:
            logger.error(f"Failed to save features to S3: {e}")
            raise
        
    def _save_transformer_to_s3(self, transformer, filename: str):
        """Save transformers to S3"""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                dump(transformer, temp_file)
                
                # Upload to S3
                s3_key = f'{self.feature_store_prefix}transformers/{filename}'
                self.s3_client.upload_file(
                    temp_file.name,
                    self.bucket_name,
                    s3_key
                )
                logger.info(f"Saved transformer to s3://{self.bucket_name}/{s3_key}")
                
            os.unlink(temp_file.name)
        
        except ClientError as e:
            logger.error(f"Failed to save transformers to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to save transformers to S3: {e}")
            raise
        
    def save_features(self, feature_df: pd.DataFrame, filename: str):
        """Save features based on environment"""
        if self.env == "local":
            self._save_features_locally(feature_df, filename)
        else:
            self._save_features_to_s3(feature_df, filename)

    def save_transformer(self, transformer, filename: str):
        """Save transformer based on environment"""
        if self.env == "local":
            self._save_transformers_locally(transformer, filename)
        else:
            self._save_transformer_to_s3(transformer, filename)
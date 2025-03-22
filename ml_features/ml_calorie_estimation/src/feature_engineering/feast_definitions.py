from datetime import timedelta
from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource

recipe = Entity(
    name="recipe",
    value_type=ValueType.INT64,
    description="Recipe identifier"
)

recipe_source = FileSource(
    path="s3://feast-recipe-data/feature_store/recipe_features.parquet",
    timestamp_field="timestamp"
)

recipe_features = FeatureView(
    name="recipe_features",
    entities=["recipes"],
    ttl=timedelta(days=365), # Features valid for 1 year
    source=recipe_source,
    online=True,
    features=[
        Feature(name="tfidf_features", dtype=ValueType.FLOAT64_LIST),
        Feature(name="target_Calories", dtype=ValueType.FLOAT64),
        Feature(name="target_Protein", dtype=ValueType.FLOAT64),
        Feature(name="target_Fat", dtype=ValueType.FLOAT64),
        Feature(name="target_Carbohydrates_net", dtype=ValueType.FLOAT64),
    ],
)

def get_recipe_source(env: str = "local") -> FileSource:
    """Get the appropriate feature source based on environment"""
    if env == "local":
        path = "ml_features/ml_calorie_estimation/feature_store/feature_repo/data/recipe_features.parquet"
    else:
        # Use the main bucket with feature store prefix
        path = "s3://iifymate-ml-data/feature_store/recipe_features.parquet"
    
    return FileSource(
        path=path,
        timestamp_field="timestamp",
    )
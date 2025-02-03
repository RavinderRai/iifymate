from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Define data source
recipe_source = FileSource(
    path="data/recipe_features.parquet",
    timestamp_field="timestamp"
)

# Define entity (like a primary key)
recipe = Entity(
    name="recipe_id",
    value_type=ValueType.INT64,
    description="recipe identifier"
)

# Define feature view
recipe_features = FeatureView(
    name="recipe_features",
    entities=[recipe],
    ttl=timedelta(days=365),
    source=recipe_source
)

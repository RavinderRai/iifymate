import pytest
import pandas as pd

from ml_features.ml_calorie_estimation.src.data_ingestion.utils import (
    create_db_config,
    load_config
)

from ml_features.ml_calorie_estimation.src.databases.manager import DatabaseManager
from ml_features.ml_calorie_estimation.src.databases.models.raw_data import RawRecipe
from ml_features.ml_calorie_estimation.src.databases.models.clean_data import CleanRecipe
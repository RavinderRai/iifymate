import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

@pytest.fixture
def sample_recipes():
    """Load first two recipes from sample data"""
    sample_data_path = Path(__file__).parent / 'data' / 'test_raw_data.csv'
    df = pd.read_csv(sample_data_path)
    return df.head(2).to_dict('records')
import os
import sys
from pathlib import Path

# Get the root directory (where ml_features is)
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))
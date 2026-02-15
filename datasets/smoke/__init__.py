import sys
import os
from pathlib import Path

# Add this directory to Python path so 'phi' can be imported as a top-level module
# This only happens when datasets.smoke is imported (i.e., when smoke dataset is used)
# Files in this directory (like data_generation.py) also add themselves to sys.path
# when run standalone via sys.path.append(os.path.dirname(__file__))
smoke_dir = Path(__file__).parent
if str(smoke_dir) not in sys.path:
    sys.path.insert(0, str(smoke_dir))

from .smoke_base import SmokeDataset
from .finetuning_smoke import FinetuningSmokeDataset

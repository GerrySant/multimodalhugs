import sys
import os

# Ensure the package directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from .custom_datasets import *
from .data import *
from .models import *
from .utils import *
from .multilingual_seq2seq_trainer import MultiLingualSeq2SeqTrainer

# Dataset classes in multimodalhugs.data are loaded lazily via __getattr__ in
# data/__init__.py (PEP 562). Wildcard imports (from .data import *) only copy
# names already present in module.__dict__, so they do not trigger __getattr__
# and the dataset classes never land in this namespace automatically.
# The __getattr__ below forwards any dataset-class lookup to multimodalhugs.data,
# preserving the pre-existing public API: `from multimodalhugs import Pose2TextDataset`.
_DATA_DATASET_CLASSES = {
    "Pose2TextDataset",
    "Video2TextDataset",
    "SignWritingDataset",
    "BilingualText2TextDataset",
    "BilingualImage2TextDataset",
    "Features2TextDataset",
}

def __getattr__(name):
    if name in _DATA_DATASET_CLASSES:
        import multimodalhugs.data as _data
        return getattr(_data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


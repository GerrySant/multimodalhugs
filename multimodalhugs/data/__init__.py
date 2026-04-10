from .utils import *
from .dataset_configs.multimodal_mt_data_config import MultimodalDataConfig
from .datacollators.multimodal_datacollator import DataCollatorMultimodalSeq2Seq

# Dataset classes are loaded lazily so that importing multimodalhugs (or
# multimodalhugs.data) does not execute modality-specific dataset modules —
# and therefore does not require their optional dependencies — unless that
# dataset class is explicitly accessed.
_LAZY_DATASETS = {
    "Pose2TextDataset":           ".datasets.pose2text",
    "Video2TextDataset":          ".datasets.video2text",
    "SignWritingDataset":          ".datasets.signwriting",
    "BilingualText2TextDataset":  ".datasets.bilingual_text2text",
    "BilingualImage2TextDataset": ".datasets.bilingual_image2text",
    "Features2TextDataset":       ".datasets.features2text",
}


def __getattr__(name):
    if name in _LAZY_DATASETS:
        import importlib
        module = importlib.import_module(_LAZY_DATASETS[name], package=__name__)
        obj = getattr(module, name)
        # Cache in module namespace so subsequent accesses skip __getattr__
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

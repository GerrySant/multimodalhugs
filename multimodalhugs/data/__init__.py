from .utils import *
from .dataset_configs.multimodal_mt_data_config import MultimodalDataConfig
from .datacollators.multimodal_datacollator import DataCollatorMultimodalSeq2Seq

# `__all__` lists only the names that are intentionally part of the public
# API and are eagerly available after `from multimodalhugs.data import *`.
# The six dataset builder classes (Pose2TextDataset, Video2TextDataset, …)
# are deliberately excluded: they are loaded lazily via __getattr__ below so
# that importing this package does not require their optional dependencies.
__all__ = [
    # dataset config / collator
    "MultimodalDataConfig",
    "DataCollatorMultimodalSeq2Seq",
    # utilities re-exported from .utils
    "BICUBIC",
    "string_to_list",
    "pad_and_create_mask",
    "center_image_on_white_background",
    "grayscale_image",
    "resize_and_center_image",
    "check_columns",
    "contains_empty",
    "sample_signal_exists",
    "file_exists_filter",
    "duration_filter",
    "split_sentence",
    "create_image",
    "normalize_images",
    "make_image_array",
    "get_images",
    "gather_appropriate_data_cfg",
    "get_all_dataclass_fields",
    "build_merged_omegaconf_config",
    "resolve_and_update_config",
]

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

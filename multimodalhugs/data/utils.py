import re
import os
import ast
import json
import torch
import pyarrow
import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from types import SimpleNamespace
from typing import List, Any, Dict, Type, Optional, Set
from PIL import Image, ImageOps, ImageDraw, ImageFont
from dataclasses import fields, is_dataclass
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _transform(n_px, mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
               std: List[float] = [0.26862954, 0.26130258, 0.27577711]):
    mean = tuple(mean)
    std = tuple(std)
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize(mean, std),
    ])

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None

def pad_and_create_mask(tensor_list):
    max_frames = max(tensor.size(0) for tensor in tensor_list)
    other_dimensions = tensor_list[0].shape[1:]
    BATCH_SIZE = len(tensor_list)
    padded_shape = (BATCH_SIZE, max_frames) + other_dimensions
    padded_tensor = torch.zeros(padded_shape)
    mask = torch.zeros((BATCH_SIZE, max_frames), dtype=torch.int)
    for i, tensor in enumerate(tensor_list):
        num_frames = tensor.size(0)
        padded_tensor[i, :num_frames] = tensor
        mask[i, :num_frames] = 1
    return padded_tensor, mask

def center_image_on_white_background(original_image, target_width=256, target_height=256, edge_gap=5):
    new_image = Image.new("RGB", (target_width, target_height), "white")
    max_width = target_width - 2 * edge_gap
    max_height = target_height - 2 * edge_gap
    original_aspect = original_image.width / original_image.height
    if original_aspect > 1:
        new_width = min(original_image.width, max_width)
        new_height = int(new_width / original_aspect)
    else:
        new_height = min(original_image.height, max_height)
        new_width = int(new_height * original_aspect)
    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    new_image.paste(resized_original, (x, y), resized_original)
    return new_image

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def grayscale_image(image):
    image_rgb = Image.new('RGB', image.size, (255, 255, 255))
    image_rgb.paste(image, (0, 0), image)
    image_l = image_rgb.convert('L')
    return image_l

def resize_and_center_image(original_image, target_width=256, target_height=256):
    original_width, original_height = original_image.size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    try:
        resized_image = original_image.resize((resized_width, resized_height), Image.ANTIALIAS)
    except AttributeError:
        resized_image = original_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    new_image = Image.new("L", (target_width, target_height), 255)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    new_image.paste(resized_image, (x, y))
    return new_image

def check_columns(dataset, required_columns):
    if isinstance(dataset, pd.DataFrame):
        return all(column in dataset.columns for column in required_columns)
    else:
        return all(column in dataset.column_names for column in required_columns)

def contains_empty(sample):
    return any(v == "" or v is None for v in sample.values())

def sample_signal_exists(sample):
    return any(v == "" or v is None for v in sample.values())

def file_exists_filter(column_name, sample):
    return os.path.exists(sample[column_name])

def duration_filter(max_frames, sample):
    return sample["DURATION"] <= max_frames

def split_sentence(sentence):
    if isinstance(sentence, pyarrow.lib.StringScalar):
        sentence = sentence.as_py()
    tokens = re.split(r'(\s+|[.,?\!”":;]+)', sentence)
    return [token for token in tokens if token.strip()]

def create_image(word, font_path, img_size=(224, 224), font_size=48):
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img_size[0] - text_width) / 2
    y = (img_size[1] - text_height) / 2
    draw.text((x, y), word, fill="black", font=font)
    return np.array(img)

def normalize_images(images, mean, std):
    normalized_images = []
    for image in images:
        normalized_image = (image / 255.0 - mean) / std
        normalized_images.append(normalized_image)
    return np.array(normalized_images)

def make_image_array(words, font_path, width, height, normalize_image, mean, std):
    images = [create_image(word=word, font_path=font_path, img_size=(width, height)) for word in words]
    images = np.array(images)
    if normalize_image:
        return normalize_images(images, mean, std)
    return images

def get_images(src_text, font_path, width, height, normalize_image, mean, std):
    words = split_sentence(src_text)
    images = make_image_array(words, font_path, width, height, normalize_image, mean, std)
    return np.transpose(images, (0, 3, 1, 2)).astype(np.float32)

def gather_appropriate_data_cfg(cfg: Any) -> Any:
    """
    Extract the relevant data configuration from the given cfg.

    Prioritizes the following structure:
    1. cfg.data if it exists
    2. cfg.dataset if it exists
    3. cfg itself if neither data nor dataset is found

    Works with OmegaConf, argparse.Namespace, or dict-like objects.

    Args:
        cfg: A configuration object (e.g., OmegaConf DictConfig, dict, or SimpleNamespace)

    Returns:
        The most appropriate sub-config to use as the data configuration.
    """
    if cfg is None:
        return {}

    # Try attribute access first
    for attr_name in ['data', 'dataset']:
        if hasattr(cfg, attr_name):
            return getattr(cfg, attr_name)

    # Fallback for dictionary-style configs
    if isinstance(cfg, dict):
        for key in ['data', 'dataset']:
            if key in cfg:
                return cfg[key]

    return cfg  # Final fallback

def get_all_dataclass_fields(cls: Type) -> Set[str]
    """
    Recursively extract all dataclass field names from a class and its base classes.
    """
    if not is_dataclass(cls):
        return set()
    result = {f.name for f in fields(cls)}
    for base in cls.__bases__:
        result |= get_all_dataclass_fields(base)
    return result

def build_merged_omegaconf_config(
    cls: Type,
    cfg: Any = None,
    **overrides
) -> tuple[Dict[str, Any], Dict[str, Any], Any]:
    """
    Build a merged OmegaConf from `cfg` and `**overrides`, keeping only valid keys for `cls`.

    Returns:
    - valid_config_dict: dict of filtered args passed to OmegaConf
    - extra_args_dict: dict of unused keys
    - OmegaConf object ready for passing to `super().__init__(cfg=...)`
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if cfg else {}
    merged_dict = {**cfg_dict, **overrides}

    valid_keys = get_all_dataclass_fields(cls)
    valid_config = {k: v for k, v in merged_dict.items() if k in valid_keys}
    extra_args = {k: v for k, v in merged_dict.items() if k not in valid_keys}

    omega_cfg = OmegaConf.create(valid_config)
    return valid_config, extra_args, omega_cfg

def resolve_and_update_config(
    config_class: Type, 
    config: Optional[object] = None, 
    kwargs: dict = None
):
    """
    Utility to resolve and update a dataclass-based config object.

    Args:
        config_class (Type): The dataclass type (e.g., Features2TextDataConfig).
        config (Optional[object]): An existing config object, or None.
        kwargs (dict): The kwargs passed to the constructor.

    Returns:
        tuple: (config_object, remaining_kwargs)
    """
    if kwargs is None:
        kwargs = {}

    # Extract config-relevant keys from kwargs
    valid_keys = {f.name for f in fields(config_class)}
    config_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid_keys}

    # Build or update the config
    if config is None:
        config = config_class(**config_kwargs)
    else:
        for k, v in config_kwargs.items():
            setattr(config, k, v)

    return config, kwargs

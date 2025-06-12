'''
multimodalhugs/utils/training_setup.py

Common utilities to initialize dataset, processor, and model for all modalities.
'''
import os
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from typing import Optional

from multimodalhugs.utils.registry import get_model_class
from multimodalhugs.utils.utils import add_argument_to_the_config, reformat_yaml_file
from multimodalhugs.utils.tokenizer_utils import extend_tokenizer


def load_config(config_path: str):
    """Load OmegaConf configuration."""
    return OmegaConf.load(config_path)


def prepare_dataset(dataset_cls, data_config, output_dir: str):
    """
    Instantiate dataset, download/prepare if needed, save to disk.
    Returns path to dataset.
    """
    dataset = dataset_cls(config=data_config)
    if getattr(data_config, 'dataset_dir', None) and os.path.exists(data_config.dataset_dir):
        return data_config.dataset_dir

    data_path = Path(output_dir) / "datasets" / dataset.name
    if not data_path.exists():
        dataset.download_and_prepare(data_path)
        dataset.as_dataset().save_to_disk(data_path)
    return str(data_path)


def load_tokenizers(data_config, output_dir: str, run_name: str):
    """
    Load pretrained tokenizer, extend vocabulary, return (tokenizer, pretrained_tokenizer, new_tokens).
    """
    pretrained = AutoTokenizer.from_pretrained(data_config.text_tokenizer_path)
    tokenizer, new_tokens = extend_tokenizer(
        data_config,
        training_output_dir=output_dir,
        model_name=run_name
    )
    return tokenizer, pretrained, new_tokens


def save_processor(processor, output_dir: str):
    path = os.path.join(output_dir, processor.name)
    processor.save_pretrained(save_directory=path, push_to_hub=False)
    return path


def build_and_save_model(model_type: str, config_path: str, tokenizer, pretrained_tokenizer, new_tokens, model_cfg: dict, output_dir: str, run_name: str):
    """
    Instantiate model via registry, save to output_dir/run_name, return path.
    """
    model_cls = get_model_class(model_type)
    kwargs = dict(
        src_tokenizer=tokenizer,
        tgt_tokenizer=pretrained_tokenizer,
        config_path=config_path,
        new_vocab_tokens=new_tokens,
        **model_cfg
    )
    model = model_cls.build_model(**kwargs)
    model_path = os.path.join(output_dir, run_name)
    model.save_pretrained(model_path)
    return model_path


def update_configs(config_path: str, processor_path: Optional[str] = None, data_path: Optional[str] = None, model_path: Optional[str] = None):
    """
    Write processor, data, and model paths back into config and reformat file.
    """
    if processor_path is not None:
        add_argument_to_the_config(config_path, "processor", "processor_name_or_path", processor_path)
    if data_path is not None:
        add_argument_to_the_config(config_path, "data", "dataset_dir", data_path)
    if model_path is not None:
        add_argument_to_the_config(config_path, "model", "model_name_or_path", model_path)
    reformat_yaml_file(config_path)
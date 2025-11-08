'''
multimodalhugs/utils/training_setup.py

Common utilities to initialize dataset, processor, and model for all modalities.
'''
import os, tempfile
import yaml
import logging
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from typing import Optional, Union

from multimodalhugs.utils.registry import get_model_class
from multimodalhugs.utils.utils import add_argument_to_the_config, reformat_yaml_file
from multimodalhugs.utils.tokenizer_utils import extend_tokenizer

logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """Load OmegaConf configuration."""
    return OmegaConf.load(config_path)


def _is_hf_dataset(path: Path) -> bool:
    """
    Heuristic to check whether a directory looks like
    a Hugging Face dataset stored on disk.
    Supports both `save_to_disk` style (with `data/`)
    and split-based layouts (train/validation/test + .arrow files).
    """
    if not (path.exists() and path.is_dir()):
        logger.debug(f"{path} does not exist or is not a directory.")
        return False

    dataset_info = (path / "dataset_info.json").exists()
    dataset_dict = (path / "dataset_dict.json").exists()
    has_data_dir = (path / "data").exists()
    has_split_dir = any((path / split).exists() for split in ["train", "validation", "test"])
    has_arrow = any(p.suffix == ".arrow" for p in path.iterdir() if p.is_file())

    is_dataset = dataset_info and (
        has_data_dir
        or dataset_dict
        or has_split_dir
        or has_arrow
    )

    logger.debug(
        f"Checking HF dataset at {path}: "
        f"dataset_info={dataset_info}, data_dir={has_data_dir}, "
        f"dataset_dict={dataset_dict}, split_dir={has_split_dir}, has_arrow={has_arrow} "
        f"-> {is_dataset}"
    )
    return is_dataset

def prepare_dataset(dataset_cls, data_config, output_dir: str, rebuild_from_scratch: bool = False):
    """
    Instantiate the dataset, download/prepare it if needed, and save it to disk.
    If `rebuild_from_scratch` is True, the HF cache will be ignored and the dataset
    will be rebuilt from zero (forced re-download / re-processing).
    Returns the path to the dataset.
    """
    logger.info("Initializing dataset class...")
    dataset = dataset_cls(config=data_config)

    # If the user provided an explicit dataset directory, make sure it's actually a dataset
    if getattr(data_config, "dataset_dir", None):
        dataset_dir = Path(data_config.dataset_dir)
        logger.info(f"User provided dataset_dir: {dataset_dir}")
        if _is_hf_dataset(dataset_dir):
            logger.info(f"Using existing dataset at {dataset_dir}")
            return str(dataset_dir)
        else:
            logger.warning(
                f"Provided dataset_dir {dataset_dir} does not look like a valid HF dataset. "
                "Will attempt to (re)create it."
            )

    dataset_name = dataset.name if data_config.name is None else data_config.name
    data_path = Path(output_dir) / "datasets" / dataset_name
    logger.info(f"Target dataset path: {data_path}")

    # Only consider it "already prepared" if it really looks like a HF dataset
    if _is_hf_dataset(data_path):
        logger.info(f"Dataset already prepared at {data_path}, reusing it.")
        return str(data_path)

    # If we get here, either it doesn't exist, or it's empty, or it's not a HF dataset → prepare it
    if not data_path.exists():
        logger.info(f"{data_path} does not exist. Creating directories...")
        data_path.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"{data_path} exists but is not a valid HF dataset. Preparing dataset...")

    download_kwargs = {}
    if rebuild_from_scratch:
        try:
            from datasets.utils.download_manager import DownloadMode
        except ModuleNotFoundError:
            from datasets.download.download_manager import DownloadMode
        download_kwargs["download_mode"] = DownloadMode.FORCE_REDOWNLOAD
        logger.info("Rebuilding dataset from scratch (forced re-download).")
    else:
        logger.warning(
            "Preparing dataset with cache enabled. If a cached version exists in the Hugging Face cache, "
            "it will be reused. Set `rebuild_from_scratch=True` (or, if you're using the "
            "`multimodalhugs-setup` CLI, pass `--rebuild-dataset-from-scratch`) to rebuild it from zero."
        )
        
    logger.info("Downloading and preparing dataset...")
    dataset.download_and_prepare(str(data_path), **download_kwargs)
    logger.info("Saving dataset to disk...")
    dataset.as_dataset().save_to_disk(str(data_path))
    logger.info(f"Dataset saved to {data_path}")

    return str(data_path)

def load_tokenizers(tokenizer_path, new_vocabulary, output_dir: Optional[str] = None, run_name: Optional[str] = None):
    """
    Load pretrained tokenizer, extend vocabulary, return (tokenizer, pretrained_tokenizer, new_tokens).
    """
    pretrained = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer, new_tokens = extend_tokenizer(
        tokenizer_path,
        new_vocabulary,
        training_output_dir=output_dir,
        model_name=run_name
    )
    return tokenizer, pretrained, new_tokens


def save_processor(processor, output_dir: str):
    path = os.path.join(output_dir, processor.name)
    processor.save_pretrained(save_directory=path, push_to_hub=False)
    return path


def build_and_save_model(model_type: str, config_path: str, tokenizer, pretrained_tokenizer, new_tokens, model_cfg: dict, output_dir: str, modal_name: str):
    """
    Instantiate model via registry, save to output_dir/model, return path.
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
    model_path = os.path.join(output_dir, modal_name)
    model.save_pretrained(model_path)
    return model_path

def build_and_save_model_from_init(model_type: str, config_path: str, output_dir: str, run_name: str):
    """
    Instantiate model via registry using __init__, save to output_dir/run_name, return path.
    """
    model_cls = get_model_class(model_type)
    model = model_cls(config_path=config_path)
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

def resolve_setup_paths(cfg, cli_output_dir=None):
    """
    Returns:
      - <output_dir>/setup  if output_dir's last component is not 'setup'
      - <output_dir>        if it already ends with 'setup'
    """
    setup_cfg = getattr(cfg, "setup", None)
    base_output_dir = cli_output_dir or (getattr(setup_cfg, "output_dir", None) if setup_cfg else None)
    if not base_output_dir:
        raise ValueError("Missing required 'output_dir'. Specify via --output_dir or cfg.setup.output_dir.")

    base_norm = os.path.normpath(base_output_dir)
    final_output_dir = base_norm if os.path.basename(base_norm) == "setup" else os.path.join(base_norm, "setup")
    return final_output_dir

def resolve_update_choice(cfg, cli_update_config: Optional[bool]) -> bool:
    """
    Decide whether to update the config file with created artifact paths.
    Priority: CLI (--update_config True/None) > cfg.setup.update_config (bool) > default False.
    """
    if cli_update_config is not None:
        return bool(cli_update_config)
    setup_cfg = getattr(cfg, "setup", None)
    cfg_val = getattr(setup_cfg, "update_config", None) if setup_cfg else None
    return bool(cfg_val) if isinstance(cfg_val, bool) else False

def print_artifact_summary(
    processor_path: Optional[str],
    model_path: Optional[str],
    data_path: Optional[str],
) -> None:
    """Print a concise summary of created actors."""
    def fmt(p: Optional[str]) -> str:
        if not p:
            return "-"
        p = str(Path(p).expanduser())
        home = str(Path.home())
        return "~" + p[len(home):] if p.startswith(home + "/") else p

    rows = [
        ("processor_name_or_path", processor_path),
        ("model_name_or_path",     model_path),
        ("dataset_dir",            data_path),
    ]
    key_w = max(len(k) for k, _ in rows)

    print("\nTraining actors created at:\n")
    for k, v in rows:
        print(f"\t{k:<{key_w}} : {fmt(v)}")
    print()

def save_actor_paths(final_output_dir: Union[str, Path],
                     proc_path: Union[str, Path, None] = None,
                     data_path: Union[str, Path, None] = None,
                     model_path: Union[str, Path, None] = None) -> Path:
    """Guarda los paths en final_output_dir/actors_paths.yaml con las claves requeridas."""
    final_dir = Path(final_output_dir).expanduser().resolve()
    final_dir.mkdir(parents=True, exist_ok=True)
    out_file = final_dir / "actors_paths.yaml"

    payload = {}

    if proc_path is not None:
        payload["processor_name_or_path"] = str(Path(proc_path).expanduser().resolve())
    if data_path is not None:
        payload["dataset_dir"] = str(Path(data_path).expanduser().resolve())
    if model_path is not None:
        payload["model_name_or_path"] = str(Path(model_path).expanduser().resolve())

    with out_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    return out_file
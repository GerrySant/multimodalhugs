'''
multimodalhugs/utils/training_setup.py

Common utilities to initialize dataset, processor, and model for all modalities.
'''
import os
import yaml
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from typing import Optional, Union

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
        dataset.download_and_prepare(str(data_path))
        dataset.as_dataset().save_to_disk(str(data_path))
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
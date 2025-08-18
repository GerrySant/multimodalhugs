import os
import argparse
from omegaconf import OmegaConf
from typing import Optional

from .setup_utils import (
    load_config, prepare_dataset, load_tokenizers,
    save_processor, build_and_save_model, update_configs, save_actor_paths,
    resolve_setup_paths, resolve_update_choice, print_artifact_summary
)

from multimodalhugs.data.datasets.features2text import Features2TextDataset, Features2TextDataConfig
from multimodalhugs.processors import Features2TextTranslationProcessor

def main(
    config_path: str,
    do_dataset: bool,
    do_processor: bool,
    do_model: bool,
    output_dir: Optional[str] = None,
    update_config: Optional[bool] = None,
):
    """
    Run setup steps for dataset preparation, processor instantiation, and model building.

    Args:
        config_path (str): Path to the OmegaConf YAML configuration file.
        do_dataset (bool): If True, prepare the dataset (download and save).
        do_processor (bool): If True, create and save the processing pipeline.
        do_model (bool): If True, build the model and save the weights.
        output_dir (str|None): Optional --output_dir from CLI (required if not in cfg.setup).
        update_config (bool|None): Optional --update_config from CLI. If True, write created
            artifact paths back into the config; otherwise print a summary. If not provided,
            falls back to cfg.setup.update_config; default is False.

    Behavior:
        - If none of do_dataset, do_processor, do_model are True, all three steps are performed.
        - Tokenizers are loaded as needed for both processor and model steps.
        - After each chosen step, the corresponding path is captured.
        - output_dir/run_name are resolved via resolve_setup_paths() from CLI or cfg.setup.
        - Whether to modify the YAML is controlled by resolve_update_choice().
    """
    cfg = load_config(config_path)

    # Resolve setup paths (required/optional + final folder creation logic)
    final_output_dir = resolve_setup_paths(cfg, output_dir)

    # 1) Dataset setup
    data_path = None
    if do_dataset:
        print("\nSetting Up Dataset:\n")
        # Instantiate and prepare dataset, then save to disk
        data_cfg = Features2TextDataConfig(cfg)
        data_path = prepare_dataset(
            Features2TextDataset,
            data_cfg,
            final_output_dir
        )

    # 2) Processor setup
    proc_path = None
    if do_processor:
        print("\nSetting Up Processor:\n")
        processor_cfg = getattr(cfg, "processor", None)

        text_tokenizer_path  = getattr(processor_cfg, "text_tokenizer_path", None) if processor_cfg else None
        new_vocabulary       = getattr(processor_cfg, "new_vocabulary", None) if processor_cfg else None

        processor_output_dir = final_output_dir

        # Load tokenizers (needed for both processor and model)
        tok, pre_tok, new = load_tokenizers(
            text_tokenizer_path,
            new_vocabulary,
        )

        # Instantiate processor with modality-specific args
        processor_kwargs = OmegaConf.to_container(processor_cfg, resolve=True) if processor_cfg else {}
        proc = Features2TextTranslationProcessor(
            tokenizer=tok,
            **processor_kwargs
        )
        proc_path = save_processor(proc, processor_output_dir)

    # 3) Model setup
    model_path = None
    if do_model:
        print("\nSetting Up Model:\n")
        # Ensure tokenizers are loaded if only building model
        try:
            tok, pre_tok, new
        except NameError:
            processor_cfg = getattr(cfg, "processor", None)
            tok, pre_tok, new = load_tokenizers(
                getattr(processor_cfg, "text_tokenizer_path", None) if processor_cfg else None,
                getattr(processor_cfg, "new_vocabulary", None) if processor_cfg else None,
            )

        # Convert OmegaConf to primitive dict for model constructor
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        mtype = cfg.model.get("type")
        model_path = build_and_save_model(
            model_type=mtype,
            config_path=config_path,
            tokenizer=tok,
            pretrained_tokenizer=pre_tok,
            new_tokens=new,
            model_cfg=model_cfg,
            output_dir=final_output_dir,
            modal_name="model"
        )

    # 4) Update config file or print summary based on the new toggle
    should_update = resolve_update_choice(cfg, update_config)
    if should_update:
        update_configs(
            config_path,
            processor_path=proc_path,
            data_path=data_path,
            model_path=model_path
        )
    else:
        print_artifact_summary(proc_path, model_path, data_path)

    # 5) Always save YAML with normalized paths
    save_actor_paths(final_output_dir, proc_path, data_path, model_path)
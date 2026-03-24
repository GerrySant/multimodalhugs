import os
import argparse
from omegaconf import OmegaConf
from typing import Optional

from .setup_utils import (
    load_config, prepare_dataset, load_tokenizers,
    save_processor, build_and_save_model, update_configs, save_actor_paths,
    resolve_setup_paths, resolve_update_choice, print_artifact_summary,
    build_processor_from_config,
)

from multimodalhugs.data.datasets.bilingual_text2text import BilingualText2TextDataset, BilingualText2textMTDataConfig
from multimodalhugs.processors import (
    MultimodalMetaProcessor,
    ProcessorSlot,
    TextModalityProcessor,
)

def main(
    config_path: str,
    do_dataset: bool,
    do_processor: bool,
    do_model: bool,
    output_dir: Optional[str] = None,
    update_config: Optional[bool] = None,
    rebuild_dataset_from_scratch: bool = False,
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
        rebuild_dataset_from_scratch (bool): If True, ignore HF cache and rebuild the dataset
            from zero (i.e. force re-download / re-processing).

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
        data_cfg = BilingualText2textMTDataConfig(cfg)
        data_path = prepare_dataset(
            BilingualText2TextDataset,
            data_cfg,
            final_output_dir,
            rebuild_from_scratch=rebuild_dataset_from_scratch,
        )

    # 2) Processor setup
    proc_path = None
    if do_processor:
        print("\nSetting Up Processor:\n")
        processor_cfg = getattr(cfg, "processor", None)

        text_tokenizer_path  = getattr(processor_cfg, "text_tokenizer_path", None) if processor_cfg else None
        new_vocabulary       = getattr(processor_cfg, "new_vocabulary", None) if processor_cfg else None

        processor_output_dir = final_output_dir

        # Instantiate processor — declarative slots config takes priority
        proc = build_processor_from_config(processor_cfg)
        if proc is None:
            # Hardcoded path: load/extend tokenizer, then build with fixed slots
            tok, pre_tok, new = load_tokenizers(text_tokenizer_path, new_vocabulary)
            proc = MultimodalMetaProcessor(
                slots=[
                    ProcessorSlot(
                        processor=TextModalityProcessor(tokenizer=tok, role="encoder"),
                        output_data_key="input_ids",
                        output_mask_key="attention_mask",
                    ),
                    ProcessorSlot(
                        processor=TextModalityProcessor(tokenizer=tok, role="label"),
                        output_data_key="labels",
                        is_label=True,
                        column_map={"decoder_prompt": "target_prefix", "output": "target"},
                    ),
                    ProcessorSlot(
                        processor=TextModalityProcessor(tokenizer=tok, role="encoder"),
                        output_data_key="encoder_prompt",
                        output_mask_key="encoder_prompt_length_padding_mask",
                        column_map={"encoder_prompt": "signal"},
                    ),
                    ProcessorSlot(
                        processor=TextModalityProcessor(tokenizer=tok, role="prompt"),
                        output_data_key="decoder_input_ids",
                        output_mask_key="decoder_attention_mask",
                        column_map={"decoder_prompt": "signal"},
                    ),
                ],
                tokenizer=tok,
            )
        else:
            # Declarative path: each TextModalityProcessor owns its own tokenizer
            # extension via new_vocabulary in its processor_kwargs. Derive tok,
            # pre_tok, and new_tokens from the first text slot that performed extension.
            # TODO: this derivation is a temporary bridge until model construction is
            # refactored to read new_tokens from the processor directly. At that point
            # this block (and load_tokenizers in the declarative path) can be removed.
            text_slot = next(
                (s for s in proc.slots if hasattr(s.processor, "new_tokens")),
                None,
            )
            if text_slot is not None:
                tok = proc.tokenizer
                pre_tok = text_slot.processor.pretrained_tokenizer
                new = text_slot.processor.new_tokens
            else:
                # TODO: non-text-output tasks — tokenizer source for model construction
                # needs a separate design (e.g. model-level tokenizer_path config key).
                tok, pre_tok, new = None, None, []
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

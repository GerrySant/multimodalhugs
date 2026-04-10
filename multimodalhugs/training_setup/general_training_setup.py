"""
multimodalhugs/training_setup/general_training_setup.py

General (modality-agnostic) training setup.

Unlike the six legacy task-specific setup files, this module does not
hardcode a dataset or processor type.  Instead:

- The dataset class is selected from ``data.dataset_type`` in the YAML config.
- The processor is built via ``build_processor_from_config`` from either the
  ``pipeline:`` shorthand or the ``slots:`` full format.  A missing or
  incomplete processor config raises an informative error rather than silently
  falling back to a hardcoded layout.

Usage (CLI):
    multimodalhugs-setup --config_path config.yaml --output_dir /out

    (no --modality argument required)

Usage (Python):
    from multimodalhugs.training_setup.general_training_setup import main
    main(config_path="config.yaml", do_dataset=True, do_processor=True, do_model=True)
"""

import logging
from omegaconf import OmegaConf
from typing import Optional

from .setup_utils import (
    load_config,
    prepare_dataset,
    load_tokenizers,
    save_processor,
    build_and_save_model,
    update_configs,
    save_actor_paths,
    resolve_setup_paths,
    resolve_update_choice,
    print_artifact_summary,
    build_processor_from_config,
    extract_tokenizer_info_from_processor_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset type → lazy import spec.
#
# Each entry maps a dataset_type string to a (module_path, class_names) tuple
# where class_names is (DatasetClassName, DataConfigClassName).  The actual
# import only happens when the specific dataset_type is requested, so running
# setup for one modality never imports optional dependencies of another.
# ---------------------------------------------------------------------------

_DATASET_IMPORT_MAP: dict = {
    "pose2text": (
        "multimodalhugs.data.datasets.pose2text",
        ("Pose2TextDataset", "Pose2TextDataConfig"),
    ),
    "video2text": (
        "multimodalhugs.data.datasets.video2text",
        ("Video2TextDataset", "Video2TextDataConfig"),
    ),
    "features2text": (
        "multimodalhugs.data.datasets.features2text",
        ("Features2TextDataset", "Features2TextDataConfig"),
    ),
    "signwriting2text": (
        "multimodalhugs.data.datasets.signwriting",
        # SignWritingDataset uses the base MultimodalDataConfig (no subclass needed).
        ("SignWritingDataset", None),
    ),
    "image2text": (
        "multimodalhugs.data.datasets.bilingual_image2text",
        ("BilingualImage2TextDataset", "BilingualImage2textMTDataConfig"),
    ),
    "text2text": (
        "multimodalhugs.data.datasets.bilingual_text2text",
        ("BilingualText2TextDataset", "BilingualText2textMTDataConfig"),
    ),
}


def _load_dataset_classes(dataset_type: str):
    """Import and return (DatasetClass, DataConfigClass) for the given dataset_type."""
    import importlib
    from multimodalhugs.data.dataset_configs.multimodal_mt_data_config import MultimodalDataConfig

    module_path, (dataset_cls_name, config_cls_name) = _DATASET_IMPORT_MAP[dataset_type]
    module = importlib.import_module(module_path)
    DatasetClass = getattr(module, dataset_cls_name)
    DataConfigClass = getattr(module, config_cls_name) if config_cls_name else MultimodalDataConfig
    return DatasetClass, DataConfigClass


def _build_dataset_map() -> dict:
    """Return the full mapping from dataset_type string to (DatasetClass, DataConfigClass).

    Imports each dataset module on demand so that optional dependencies for
    unused modalities are never required.  Kept as a public helper for tests
    and tooling that need to inspect the complete map at once.
    """
    return {dtype: _load_dataset_classes(dtype) for dtype in _DATASET_IMPORT_MAP}


def main(
    config_path: str,
    do_dataset: bool,
    do_processor: bool,
    do_model: bool,
    output_dir: Optional[str] = None,
    update_config: Optional[bool] = None,
    rebuild_dataset_from_scratch: bool = False,
    default_dataset_type: Optional[str] = None,
):
    """
    Run the general (modality-agnostic) setup pipeline.

    Args:
        config_path: Path to the OmegaConf YAML configuration file.
        do_dataset: If True, prepare and save the dataset.
        do_processor: If True, build and save the processor.
        do_model: If True, build and save the model weights.
        output_dir: Optional output directory override (required if not set
            via ``cfg.setup.output_dir``).
        update_config: If True, write created artifact paths back into the
            config YAML.  Falls back to ``cfg.setup.update_config`` (default
            False).
        rebuild_dataset_from_scratch: If True, ignore the HuggingFace cache
            and rebuild the dataset from zero.
        default_dataset_type: Fallback ``dataset_type`` used when
            ``data.dataset_type`` is absent from the config.  The legacy
            task-specific setup scripts pass this to stay backward-compatible
            with configs that predate the ``dataset_type`` field.

    Raises:
        ValueError: If ``do_dataset=True`` and ``data.dataset_type`` is absent
            (and no ``default_dataset_type`` was given) or unknown.
        ValueError: If ``do_processor=True`` and the processor config has
            neither ``pipeline:`` nor ``slots:``.
        ValueError: If ``do_model=True`` without a prior ``do_processor=True``
            step and no tokenizer path can be found in the config.
    """
    cfg = load_config(config_path)
    final_output_dir = resolve_setup_paths(cfg, output_dir)

    # Tokenizer variables are set in the processor step when do_processor=True,
    # or derived from the config in the model step when do_processor=False.
    tok = pre_tok = None
    new: list = []

    # ------------------------------------------------------------------
    # 1. Dataset setup
    # ------------------------------------------------------------------
    data_path = None
    if do_dataset:
        print("\nSetting Up Dataset:\n")

        data_cfg_node = getattr(cfg, "data", None)
        dataset_type = getattr(data_cfg_node, "dataset_type", None) if data_cfg_node else None
        if not dataset_type:
            dataset_type = default_dataset_type

        if not dataset_type:
            raise ValueError(
                "data.dataset_type is required for the general setup path. "
                "Add 'dataset_type: <type>' under the 'data:' section of your config. "
                f"Supported values: {sorted(_DATASET_IMPORT_MAP)}."
            )

        if dataset_type not in _DATASET_IMPORT_MAP:
            raise ValueError(
                f"Unknown data.dataset_type: '{dataset_type}'. "
                f"Supported values: {sorted(_DATASET_IMPORT_MAP)}."
            )

        DatasetClass, DataConfigClass = _load_dataset_classes(dataset_type)
        data_cfg = DataConfigClass(cfg)
        data_path = prepare_dataset(
            DatasetClass,
            data_cfg,
            final_output_dir,
            rebuild_from_scratch=rebuild_dataset_from_scratch,
        )

    # ------------------------------------------------------------------
    # 2. Processor setup
    # ------------------------------------------------------------------
    proc_path = None
    if do_processor:
        print("\nSetting Up Processor:\n")

        processor_cfg = getattr(cfg, "processor", None)
        proc = build_processor_from_config(processor_cfg)

        if proc is None:
            raise ValueError(
                "The general setup path requires a declarative processor config. "
                "Add either 'pipeline:' (shorthand) or 'slots:' (full format) under "
                "the 'processor:' section of your config. "
                "See docs/processors/processor_config_formats.md for details. "
                "To use the legacy hardcoded processor, pass --modality <name> instead."
            )

        # Extract tokenizer info from the built processor so the model
        # construction step has access to tok, pre_tok, and new_tokens.
        # TextModalityProcessor exposes these as bridge attributes after
        # loading and extending the tokenizer in its __init__.
        text_slot = next(
            (s for s in proc.slots if hasattr(s.processor, "new_tokens")),
            None,
        )
        if text_slot is not None:
            tok = proc.tokenizer
            pre_tok = text_slot.processor.pretrained_tokenizer
            new = text_slot.processor.new_tokens
        else:
            # No text slots (e.g. image→pose pipeline); tokenizer is not needed
            # unless the model itself requires one.
            tok, pre_tok, new = None, None, []

        proc_path = save_processor(proc, final_output_dir)

    # ------------------------------------------------------------------
    # 3. Model setup
    # ------------------------------------------------------------------
    model_path = None
    if do_model:
        print("\nSetting Up Model:\n")

        if tok is None:
            # do_model=True without do_processor=True: reconstruct tokenizers
            # directly from the config without building the processor.
            processor_cfg = getattr(cfg, "processor", None)
            tok_path, new_vocab = extract_tokenizer_info_from_processor_config(processor_cfg)
            if tok_path is None:
                raise ValueError(
                    "Cannot determine tokenizer_path for model construction. "
                    "Ensure your config contains 'tokenizer_path' in either the "
                    "'pipeline:' shorthand or in a text slot's 'processor_kwargs'. "
                    "Alternatively, run with do_processor=True so the tokenizer is "
                    "derived from the built processor."
                )
            tok, pre_tok, new = load_tokenizers(tok_path, new_vocab)

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
            modal_name="model",
        )

    # ------------------------------------------------------------------
    # 4. Persist artifact paths
    # ------------------------------------------------------------------
    should_update = resolve_update_choice(cfg, update_config)
    if should_update:
        update_configs(
            config_path,
            processor_path=proc_path,
            data_path=data_path,
            model_path=model_path,
        )
    else:
        print_artifact_summary(proc_path, model_path, data_path)

    save_actor_paths(final_output_dir, proc_path, data_path, model_path)

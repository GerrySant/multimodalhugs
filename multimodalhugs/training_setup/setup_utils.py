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


# ---------------------------------------------------------------------------
# Pipeline shorthand — preset definitions
# ---------------------------------------------------------------------------

# Mapping from pipeline name to the default template for the *modality slot*
# (first slot in the generated list).  The three standard text output slots
# are always appended afterwards (see _TEXT_SLOTS_TEMPLATE below).
#
# Each entry is a plain dict with the following keys:
#   processor_class  (str)  — name of the ModalityProcessor subclass
#   output_data_key  (str)  — key under which the slot writes its tensor
#   output_mask_key  (str)  — key for the padding mask tensor
#   column_map       (dict) — mapping from TSV column names to processor params
#   _needs_text_kwargs (bool, optional) — internal flag: when True the
#       expansion injects tokenizer_path, new_vocabulary, and role=input into
#       processor_kwargs (used for the text2text source slot).
_PIPELINE_PRESETS: dict = {
    # --- Temporal / sequence modalities (have start/end offsets in the TSV) ---
    "pose2text": {
        "processor_class": "PoseModalityProcessor",
        "output_data_key": "input_frames",
        "output_mask_key": "attention_mask",
        "column_map": {
            "signal": "signal",
            "signal_start": "signal_start",
            "signal_end": "signal_end",
        },
    },
    "video2text": {
        "processor_class": "VideoModalityProcessor",
        "output_data_key": "input_frames",
        "output_mask_key": "attention_mask",
        "column_map": {
            "signal": "signal",
            "signal_start": "signal_start",
            "signal_end": "signal_end",
        },
    },
    "features2text": {
        "processor_class": "FeaturesModalityProcessor",
        "output_data_key": "input_frames",
        "output_mask_key": "attention_mask",
        "column_map": {
            "signal": "signal",
            "signal_start": "signal_start",
            "signal_end": "signal_end",
        },
    },
    # --- Frame / image modalities (no temporal offsets) ---
    "image2text": {
        "processor_class": "ImageModalityProcessor",
        "output_data_key": "input_frames",
        "output_mask_key": "attention_mask",
        "column_map": {"signal": "signal"},
    },
    "signwriting2text": {
        "processor_class": "SignwritingModalityProcessor",
        "output_data_key": "input_frames",
        "output_mask_key": "attention_mask",
        "column_map": {"signal": "signal"},
    },
    # --- Text source modality ---
    # The source text is processed by a TextModalityProcessor (role=input),
    # so it also needs tokenizer_path / new_vocabulary injected at expansion
    # time (signalled by _needs_text_kwargs=True).
    "text2text": {
        "processor_class": "TextModalityProcessor",
        "output_data_key": "input_frames",
        "output_mask_key": "attention_mask",
        "column_map": {"signal": "signal"},
        "_needs_text_kwargs": True,
    },
}

# Three standard text output slots appended to every pipeline expansion.
# They handle encoder prompt, decoder prompt (input side), and target labels.
# tokenizer_path, new_vocabulary, and role are filled in at expansion time.
# Private keys prefixed with "_" are stripped before the slot dict is used.
_TEXT_SLOTS_TEMPLATE: list = [
    # 1. Target labels slot — reads (decoder_prompt, output) → (target_prefix, target)
    #    The internal param names differ from the TSV column names; the
    #    column_map absorbs the rename so the TSV schema is unchanged.
    {
        "processor_class": "TextModalityProcessor",
        "output_data_key": "labels",
        "is_label": True,
        "column_map": {"decoder_prompt": "target_prefix", "output": "target"},
        "_role": "target",
    },
    # 2. Encoder prompt slot — reads encoder_prompt TSV column as plain input text
    {
        "processor_class": "TextModalityProcessor",
        "output_data_key": "encoder_prompt",
        "output_mask_key": "encoder_prompt_length_padding_mask",
        "column_map": {"encoder_prompt": "signal"},
        "_role": "input",
    },
    # 3. Decoder input slot — reads decoder_prompt TSV column as decoder context
    {
        "processor_class": "TextModalityProcessor",
        "output_data_key": "decoder_input_ids",
        "output_mask_key": "decoder_attention_mask",
        "column_map": {"decoder_prompt": "signal"},
        "_role": "input",
    },
]


def expand_pipeline_shorthand(processor_cfg):
    """
    Normalize a compact ``pipeline:`` processor config into a full ``slots:`` list.

    This is a *pure normalization step* — it does not instantiate any processor
    objects.  Call it before ``build_processor_from_config``, which remains
    unaware of the shorthand format and only ever sees ``slots:``.

    Three config levels are supported transparently:

    1. **Shorthand** (``pipeline:`` key present)
       A compact, human-friendly declaration.  This function expands it into
       the full ``slots:`` representation on the fly.

    2. **Full slots** (``slots:`` key present, no ``pipeline:`` key)
       Passed through unchanged — this function is a no-op.

    3. **Legacy / neither key**
       Also passed through unchanged so the caller can fall back to its
       existing hardcoded construction path.

    Shorthand YAML format
    ---------------------
    ::

        processor:
          pipeline: video2text            # required — one of the six supported values
          tokenizer_path: facebook/m2m100_418M  # required — shared by all text slots
          new_vocabulary: "__asl__"       # optional — comma-separated tokens or path
          modality_kwargs:                # optional — forwarded to the modality slot's
            skip_frames_stride: 2        #   processor_kwargs unchanged
          slot_overrides:                 # optional — sparse per-slot overrides
            encoder_prompt:              #   keyed by output_data_key
              column_map:
                my_column: signal

    Supported ``pipeline`` values: ``pose2text``, ``video2text``,
    ``image2text``, ``features2text``, ``signwriting2text``, ``text2text``.

    The modality slot is always placed first; the three standard text output
    slots (``labels``, ``encoder_prompt``, ``decoder_input_ids``) follow in
    that order.

    ``slot_overrides``
    ------------------
    Each key in ``slot_overrides`` must match an ``output_data_key`` produced
    by the pipeline template (e.g. ``input_frames``, ``labels``,
    ``encoder_prompt``, ``decoder_input_ids``).  The value is a dict of
    top-level slot fields to *merge* into the generated slot.  Merging rules:

    - Scalar fields (``output_mask_key``, ``is_label``) are replaced outright.
    - Dict fields (``column_map``, ``processor_kwargs``) are shallow-merged so
      only the specified keys are overridden; unmentioned keys are preserved.

    Override keys that do not match any generated slot produce a warning and
    are silently ignored (rather than raising an error) to make configs
    resilient to minor pipeline changes.

    Full ``slots:`` format is always accepted
    -----------------------------------------
    Users who need full control can skip the shorthand entirely::

        processor:
          slots:
            - processor_class: VideoModalityProcessor
              output_data_key: input_frames
              ...

    Args:
        processor_cfg: An OmegaConf DictConfig or plain dict representing
            ``cfg.processor``.  When ``pipeline:`` is present, the function
            returns a new config of the same type with ``slots:`` populated
            and the shorthand keys removed.  When the input has neither
            ``pipeline:`` nor ``slots:``, it is returned unchanged.

    Returns:
        The (possibly expanded) config.  When ``pipeline:`` was present the
        return value contains ``slots:`` and no longer contains ``pipeline:``,
        ``modality_kwargs:``, or ``slot_overrides:``.

    Raises:
        ValueError: If ``pipeline`` names an unsupported value.
        ValueError: If ``tokenizer_path`` is absent when ``pipeline:`` is used.
    """
    # ------------------------------------------------------------------
    # 1. Normalise to a plain Python dict for uniform handling.
    # ------------------------------------------------------------------
    is_omegaconf = OmegaConf.is_config(processor_cfg)
    if is_omegaconf:
        cfg_dict = OmegaConf.to_container(processor_cfg, resolve=True)
    elif processor_cfg is None:
        return processor_cfg
    else:
        cfg_dict = dict(processor_cfg)

    # ------------------------------------------------------------------
    # 2. Passthrough cases — nothing to expand.
    # ------------------------------------------------------------------
    if "pipeline" not in cfg_dict:
        # Either full slots:, legacy config, or None — caller handles it.
        return processor_cfg

    # ------------------------------------------------------------------
    # 3. Validate required shorthand fields.
    # ------------------------------------------------------------------
    pipeline = cfg_dict["pipeline"]
    if pipeline not in _PIPELINE_PRESETS:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'. "
            f"Supported values: {sorted(_PIPELINE_PRESETS)}."
        )

    tokenizer_path = cfg_dict.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError(
            "'tokenizer_path' is required when using the 'pipeline' shorthand."
        )

    # Optional top-level shorthand fields.
    new_vocabulary = cfg_dict.get("new_vocabulary")       # str | None
    modality_kwargs = dict(cfg_dict.get("modality_kwargs") or {})
    slot_overrides = dict(cfg_dict.get("slot_overrides") or {})

    # ------------------------------------------------------------------
    # 4. Build the modality slot (always first in the generated list).
    # ------------------------------------------------------------------
    preset = dict(_PIPELINE_PRESETS[pipeline])  # shallow copy — do not mutate global
    needs_text_kwargs = preset.pop("_needs_text_kwargs", False)

    # Start from the modality_kwargs provided by the user and, for the
    # text2text pipeline, also inject the shared tokenizer settings and
    # role so that the source TextModalityProcessor is fully configured.
    modality_proc_kwargs = dict(modality_kwargs)
    if needs_text_kwargs:
        modality_proc_kwargs["tokenizer_path"] = tokenizer_path
        if new_vocabulary:
            modality_proc_kwargs["new_vocabulary"] = new_vocabulary
        # role=input distinguishes the source slot from target (label) slots.
        modality_proc_kwargs["role"] = "input"

    modality_slot: dict = {
        "processor_class": preset["processor_class"],
        "output_data_key": preset["output_data_key"],
        "output_mask_key": preset["output_mask_key"],
        "column_map": dict(preset["column_map"]),
    }
    if modality_proc_kwargs:
        modality_slot["processor_kwargs"] = modality_proc_kwargs

    # ------------------------------------------------------------------
    # 5. Build the three standard text output slots.
    # ------------------------------------------------------------------
    slots: list = [modality_slot]

    for tmpl in _TEXT_SLOTS_TEMPLATE:
        # Strip internal bookkeeping keys (prefixed with "_") before use.
        role = tmpl["_role"]

        text_proc_kwargs: dict = {"tokenizer_path": tokenizer_path, "role": role}
        if new_vocabulary:
            text_proc_kwargs["new_vocabulary"] = new_vocabulary

        slot: dict = {
            "processor_class": tmpl["processor_class"],
            "output_data_key": tmpl["output_data_key"],
            "column_map": dict(tmpl["column_map"]),
            "processor_kwargs": text_proc_kwargs,
        }
        # Carry over optional fields only when they are present in the template.
        if "output_mask_key" in tmpl:
            slot["output_mask_key"] = tmpl["output_mask_key"]
        if tmpl.get("is_label"):
            slot["is_label"] = True

        slots.append(slot)

    # ------------------------------------------------------------------
    # 6. Apply slot_overrides — sparse per-slot customisation.
    # ------------------------------------------------------------------
    if slot_overrides:
        # Index the generated slots by output_data_key for O(1) lookup.
        slots_by_key = {s["output_data_key"]: s for s in slots}

        for override_key, override_fields in slot_overrides.items():
            if override_key not in slots_by_key:
                logger.warning(
                    "slot_overrides key '%s' does not match any generated slot "
                    "(available: %s). Override ignored.",
                    override_key,
                    list(slots_by_key),
                )
                continue

            target_slot = slots_by_key[override_key]
            for field, value in override_fields.items():
                existing = target_slot.get(field)
                if isinstance(value, dict) and isinstance(existing, dict):
                    # Shallow-merge dicts (e.g. column_map, processor_kwargs)
                    # so the caller only needs to specify the changed keys.
                    target_slot[field] = {**existing, **value}
                else:
                    # Scalar fields (e.g. output_mask_key, is_label) replace.
                    target_slot[field] = value

    # ------------------------------------------------------------------
    # 7. Assemble the expanded config.
    # ------------------------------------------------------------------
    # Copy all processor-level keys that are not shorthand-specific, then
    # replace them with the expanded slots list.  This preserves any extra
    # processor-level fields the user may have added (e.g. a future
    # ``save_directory:`` key) without the shorthand keys leaking through.
    _SHORTHAND_KEYS = {"pipeline", "tokenizer_path", "new_vocabulary",
                       "modality_kwargs", "slot_overrides"}
    expanded = {k: v for k, v in cfg_dict.items() if k not in _SHORTHAND_KEYS}
    expanded["slots"] = slots

    # Return the same config type as the input so callers are not surprised.
    if is_omegaconf:
        return OmegaConf.create(expanded)
    return expanded


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


def build_processor_from_config(processor_cfg):
    """
    Build a MultimodalMetaProcessor declaratively from a YAML processor config.

    This function accepts either the **shorthand** ``pipeline:`` format or the
    **full** ``slots:`` format.  When the shorthand is present,
    ``expand_pipeline_shorthand`` is called first to normalise the config into
    the full slots representation; ``build_processor_from_config`` itself only
    ever operates on the ``slots:`` key.

    Full-slots format
    -----------------
    ``processor_cfg.slots`` must be a list of slot dicts.  Each slot dict
    must have:

      - ``processor_class`` (str): name of a ``ModalityProcessor`` subclass
        exported from ``multimodalhugs.processors``
      - ``output_data_key`` (str)
      - ``output_mask_key`` (str, optional)
      - ``column_map`` (dict, optional; defaults to ``{"signal": "signal"}``)
      - ``is_label`` (bool, optional; defaults to False)
      - ``processor_kwargs`` (dict, optional): extra kwargs forwarded to the
        processor constructor.  For ``TextModalityProcessor``, include
        ``tokenizer_path`` here — the processor loads it internally.

    Shorthand format
    ----------------
    See ``expand_pipeline_shorthand`` for the compact ``pipeline:`` syntax.
    Example::

        processor:
          pipeline: pose2text
          tokenizer_path: facebook/m2m100_418M
          new_vocabulary: "__asl__"

    The tokenizer is NOT injected at this level.  Each processor class owns
    its own constructor arguments.  ``MultimodalMetaProcessor.tokenizer`` is
    auto-derived from the first text slot for HF ProcessorMixin compatibility.

    Returns ``None`` if ``processor_cfg`` has neither a ``pipeline:`` key (after
    expansion) nor a ``slots:`` key, so the caller can fall back to its existing
    hardcoded construction path.
    """
    import multimodalhugs.processors as proc_module
    from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot

    # Normalise the shorthand pipeline: format → full slots: format.
    # This is a no-op when slots: is already present or when neither key exists.
    processor_cfg = expand_pipeline_shorthand(processor_cfg)

    if processor_cfg is None:
        return None

    # Support both OmegaConf DictConfig and plain dicts (the latter can be
    # returned by expand_pipeline_shorthand when given a plain-dict input).
    if isinstance(processor_cfg, dict):
        slot_cfgs = processor_cfg.get("slots")
    else:
        slot_cfgs = getattr(processor_cfg, "slots", None)

    if not slot_cfgs:
        return None

    # Normalise slot_cfgs to a plain Python list regardless of origin.
    if OmegaConf.is_config(slot_cfgs):
        slot_cfgs = OmegaConf.to_container(slot_cfgs, resolve=True)

    slots = []
    for slot_cfg in slot_cfgs:
        proc_cls = getattr(proc_module, slot_cfg["processor_class"])
        proc_kwargs = dict(slot_cfg.get("processor_kwargs") or {})
        proc = proc_cls(**proc_kwargs)
        slots.append(ProcessorSlot(
            processor=proc,
            output_data_key=slot_cfg["output_data_key"],
            output_mask_key=slot_cfg.get("output_mask_key"),
            column_map=slot_cfg.get("column_map", {"signal": "signal"}),
            is_label=slot_cfg.get("is_label", False),
        ))
    return MultimodalMetaProcessor(slots=slots)


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
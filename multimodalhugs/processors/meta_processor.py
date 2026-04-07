import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import PROCESSOR_NAME

from multimodalhugs.processors.modality_processor import ModalityProcessor

logger = logging.getLogger(__name__)


@dataclass
class ProcessorSlot:
    """
    Binds a ModalityProcessor to a set of dataset item fields and a set of
    forward() argument names.

    column_map  — maps dataset item field name → processor parameter name.
                  The first key is the *primary* field: its value is replaced
                  with a preprocessed tensor by _transform_get_items_output
                  when process_sample returns a tensor.
                  All other keys are context-only (e.g. temporal bounds) and
                  are passed to process_sample() but not written back.
                  Default {"signal": "signal"} covers the standard single-field
                  case and requires no explicit configuration.

    output_data_key  — the key under which the data tensor is stored in the
                       BatchFeature returned by MultimodalMetaProcessor.__call__.
    output_mask_key  — the key for the mask tensor (None if no mask is needed).
    is_label         — marks this slot as producing a loss target. Does not
                       affect processing logic; used by callers (trainers,
                       collators) to identify which output key is the label.
    """
    processor: ModalityProcessor
    output_data_key: str
    output_mask_key: Optional[str] = None
    column_map: Dict[str, str] = field(
        default_factory=lambda: {"signal": "signal"}
    )
    is_label: bool = False

    @property
    def primary_field(self) -> str:
        """The dataset item field whose value is replaced with a tensor."""
        return next(iter(self.column_map))


class MultimodalMetaProcessor(ProcessorMixin):
    """
    Orchestrates a flat list of ProcessorSlots to produce a full model batch.

    Each slot declares:
      - which dataset columns to read (column_map)
      - which ModalityProcessor to use
      - where to write the processed tensors (output_data_key / output_mask_key)
      - whether it produces a loss target (is_label)

    The MetaProcessor iterates slots in declaration order and has no knowledge
    of task structure (encoder vs. label vs. prompt). All semantic meaning lives
    in the processors and their slot configuration.

    slots     — flat list of ProcessorSlot objects in processing order.

                Ordering rules:
                  • Slots are processed strictly in declaration order during
                    both _transform_get_items_output (dataset transform) and
                    __call__ (collation).
                  • Slots are independent: no slot can read output produced by
                    an earlier slot.  Each slot reads raw dataset columns via
                    its column_map and writes to a dedicated output_data_key.
                  • Duplicate output_data_key values across slots are rejected
                    at construction time.  The skip-if-present rule in __call__
                    is intended for *external* overrides (e.g. a DataCollator
                    pre-populating decoder_input_ids), not for intra-slot
                    communication.

    tokenizer — read-only property derived from the first text slot that
                owns a tokenizer. None for pipelines with no text slots.
                save_pretrained writes the tokenizer to disk alongside the
                slot config so that from_pretrained can reconstruct text
                slots without requiring the original tokenizer_path.
    """

    # MultimodalMetaProcessor.__init__ has no modality-named parameters
    # (ProcessorMixin 5.x looks for tokenizer, image_processor, …), so
    # get_attributes() naturally returns [] and ProcessorMixin.__init__
    # requires no typed components. save_pretrained/from_pretrained are
    # fully overridden below.
    tokenizer_class = None
    name = "multimodal_meta_processor"

    def __init__(
        self,
        slots: List[ProcessorSlot],
    ):
        if not slots:
            raise ValueError(
                "MultimodalMetaProcessor requires at least one ProcessorSlot."
            )
        seen_data_keys: set = set()
        seen_mask_keys: set = set()
        for slot in slots:
            if slot.output_data_key in seen_data_keys:
                raise ValueError(
                    f"Duplicate output_data_key '{slot.output_data_key}' detected in slots. "
                    "Each slot must write to a unique key. "
                    "Pre-populating a key before __call__ (e.g. from a DataCollator) is the "
                    "intended mechanism for overrides — not duplicate slot declarations."
                )
            seen_data_keys.add(slot.output_data_key)
            if slot.output_mask_key is not None:
                if slot.output_mask_key in seen_mask_keys:
                    raise ValueError(
                        f"Duplicate output_mask_key '{slot.output_mask_key}' detected in slots. "
                        "Each slot must write its mask to a unique key."
                    )
                seen_mask_keys.add(slot.output_mask_key)
        self.slots = slots
        super().__init__()

    @property
    def tokenizer(self):
        """Return the tokenizer from the first text slot, or None if no text slot exists."""
        return next(
            (
                s.processor.tokenizer
                for s in self.slots
                if hasattr(s.processor, "tokenizer") and s.processor.tokenizer is not None
            ),
            None,
        )

    # ------------------------------------------------------------------
    # HF save / load compatibility
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, push_to_hub: bool = False, **kwargs):
        """
        Save the processor config and, when present, the tokenizer.

        Overrides ProcessorMixin.save_pretrained to handle the case where
        self.tokenizer is None (e.g. future non-text-output tasks).
        """
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, PROCESSOR_NAME)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
        if push_to_hub:
            self.push_to_hub(save_directory, **kwargs)

    @staticmethod
    def _serialize_slot(slot: "ProcessorSlot") -> Dict[str, Any]:
        """Return a JSON-serializable dict describing one ProcessorSlot."""
        proc = slot.processor
        proc_kwargs: Dict[str, Any] = {}
        # Attributes excluded from slot serialization fall into two categories:
        #   Saved separately  — tokenizer, pretrained_tokenizer: saved as a full
        #     tokenizer directory by save_pretrained; not needed in the slot JSON.
        #   Reconstructable derived  — custom_preprocessor, new_tokens: heavy
        #     runtime objects rebuilt deterministically in __init__ from their
        #     serializable counterparts (custom_preprocessor_path, tokenizer
        #     extension); they will be correctly present after from_pretrained.
        _SKIP = {"tokenizer", "pretrained_tokenizer", "new_tokens", "custom_preprocessor"}
        for k, v in proc.__dict__.items():
            if k.startswith("_") or k in _SKIP or callable(v):
                continue
            try:
                json.dumps(v)
                proc_kwargs[k] = v
            except (TypeError, ValueError):
                logger.warning(
                    "Processor attribute '%s' of type %s is not JSON-serializable "
                    "and will be omitted from the saved config. The processor will "
                    "be missing this attribute after from_pretrained().",
                    k, type(v).__name__,
                )
        return {
            "processor_class": proc.__class__.__name__,
            "processor_kwargs": proc_kwargs,
            "output_data_key": slot.output_data_key,
            "output_mask_key": slot.output_mask_key,
            "column_map": slot.column_map,
            "is_label": slot.is_label,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of this processor."""
        return {
            "processor_class": self.__class__.__name__,
            "slots": [self._serialize_slot(s) for s in self.slots],
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        processor_registry: Optional[Dict[str, type]] = None,
        **kwargs,
    ):
        """
        Reconstruct a MultimodalMetaProcessor saved with save_pretrained().

        processor_registry — optional mapping of class name → class for
            user-defined ModalityProcessor subclasses not exported from
            ``multimodalhugs.processors``.  Lookup order: registry first,
            then the built-in ``multimodalhugs.processors`` module.

            Example::

                from mylib import MyCustomProcessor
                proc = MultimodalMetaProcessor.from_pretrained(
                    "/path/to/saved",
                    processor_registry={"MyCustomProcessor": MyCustomProcessor},
                )

            Note: all processor classes used in a saved config must be either
            exported from ``multimodalhugs.processors`` or supplied via this
            argument.  See issue #77 for a planned global-registration API.
        """
        import multimodalhugs.processors as proc_module

        config, _ = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)

        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            # No tokenizer files present in the directory — valid for non-text pipelines.
            logger.debug(
                "No tokenizer found in '%s'; setting tokenizer=None. "
                "This is expected for processors with no text slots.",
                pretrained_model_name_or_path,
            )
            tokenizer = None

        # Tokenizer cache: the tokenizer saved alongside this processor is
        # already fully extended.  Pre-seed the cache with it, keyed by the
        # (tokenizer_path, new_vocabulary) of the first text slot, so that all
        # subsequent text slots with the same key can skip the redundant
        # AutoTokenizer.from_pretrained + extend_tokenizer calls that would
        # otherwise run once per slot.
        _tok_cache: Dict[tuple, Any] = {}
        if tokenizer is not None:
            for _sd in config["slots"]:
                _pkw = _sd.get("processor_kwargs", {})
                if "tokenizer_path" in _pkw:
                    _key = (_pkw.get("tokenizer_path"), _pkw.get("new_vocabulary"))
                    _tok_cache[_key] = tokenizer
                    break

        def _reconstruct_slot(slot_dict: Dict[str, Any]) -> ProcessorSlot:
            class_name = slot_dict["processor_class"]
            if processor_registry and class_name in processor_registry:
                proc_cls = processor_registry[class_name]
            else:
                try:
                    proc_cls = getattr(proc_module, class_name)
                except AttributeError:
                    raise AttributeError(
                        f"Processor class '{class_name}' not found in "
                        "multimodalhugs.processors. If this is a user-defined "
                        "subclass, pass it via processor_registry="
                        f"{{''{class_name}'': YourClass}}."
                    )
            proc_kwargs = dict(slot_dict["processor_kwargs"])
            sig = inspect.signature(proc_cls.__init__)
            if "tokenizer" in sig.parameters:
                tok_path = proc_kwargs.get("tokenizer_path")
                new_vocab = proc_kwargs.get("new_vocabulary")
                cache_key = (tok_path, new_vocab)
                if cache_key in _tok_cache:
                    # Inject the already-extended tokenizer and skip re-extension.
                    # new_vocabulary is omitted from the constructor call so that
                    # TextModalityProcessor.__init__ does not call extend_tokenizer
                    # again; it is restored on the instance afterwards so that
                    # re-serialization with save_pretrained produces the correct JSON.
                    kwargs_no_vocab = {k: v for k, v in proc_kwargs.items() if k != "new_vocabulary"}
                    proc = proc_cls(tokenizer=_tok_cache[cache_key], **kwargs_no_vocab)
                    proc.new_vocabulary = new_vocab
                else:
                    proc = proc_cls(tokenizer=tokenizer, **proc_kwargs)
                    if tok_path is not None:
                        _tok_cache[cache_key] = proc.tokenizer
            else:
                proc = proc_cls(**proc_kwargs)
            return ProcessorSlot(
                processor=proc,
                output_data_key=slot_dict["output_data_key"],
                output_mask_key=slot_dict.get("output_mask_key"),
                column_map=slot_dict["column_map"],
                is_label=slot_dict.get("is_label", False),
            )

        return cls(slots=[_reconstruct_slot(s) for s in config["slots"]])

    def __repr__(self) -> str:
        tok = type(self.tokenizer).__name__ if self.tokenizer is not None else "None"
        slots_str = ", ".join(
            f"{s.output_data_key}→{type(s.processor).__name__}" for s in self.slots
        )
        return f"MultimodalMetaProcessor(slots=[{slots_str}], tokenizer={tok})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sample_values(self, sample: Dict[str, Any], slot: "ProcessorSlot") -> Any:
        """
        Extract and optionally preprocess a value from a sample dict for a slot.

        If the primary field value is already a tensor (pre-processed by
        _transform_get_items_output), return it directly.  Otherwise build
        the input for process_sample from the column_map and call it.
        """
        primary = slot.primary_field
        value = sample[primary]
        if isinstance(value, torch.Tensor):
            return value
        if len(slot.column_map) > 1:
            values = {param: sample[col] for col, param in slot.column_map.items()}
        else:
            values = value
        return slot.processor.process_sample(values)

    # ------------------------------------------------------------------
    # Dataset-level transform  (called via dataset.with_transform)
    # ------------------------------------------------------------------

    def _transform_get_items_output(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Convert raw dataset item fields to tensors for each slot's primary field.
        Registered with dataset.with_transform() — runs before batching.

        Only writes back to the batch dict when process_sample returns a tensor,
        so that text slots (which are no-ops) do not corrupt their columns.
        """
        for slot in self.slots:
            primary = slot.primary_field
            if primary not in batch:
                logger.warning(
                    "Slot '%s': primary column '%s' not found in batch (available: %s). "
                    "Check the column_map for this slot — a typo in the dataset column "
                    "name is the most common cause.",
                    slot.output_data_key,
                    primary,
                    sorted(batch.keys()),
                )
                continue
            n = len(batch[primary])
            new_values = []
            for i in range(n):
                raw = (
                    {param: batch[col][i] for col, param in slot.column_map.items()}
                    if len(slot.column_map) > 1
                    else batch[primary][i]
                )
                result = slot.processor.process_sample(raw)
                new_values.append(result if isinstance(result, torch.Tensor) else batch[primary][i])
            batch[primary] = new_values
        return batch

    # ------------------------------------------------------------------
    # Collator-level call  (called by DataCollator with a full batch)
    # ------------------------------------------------------------------

    def __call__(
        self,
        batch: List[Dict[str, Any]],
        batch_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Process a collated batch (list of sample dicts) into a BatchFeature.

        Iterates slots in declaration order. Skips a slot if its output_data_key
        is already present in the result. This skip-if-present rule is also the
        mechanism that allows post-hoc overrides by callers: for example, the
        DataCollator can call prepare_decoder_input_ids_from_labels() and place
        the result under "decoder_input_ids" in batch_dict before invoking the
        processor, and the corresponding slot will be skipped automatically.

        batch_dict — pre-populated dict merged into the result before slot
                     processing begins. Keys already present are not overwritten.
        """
        result = dict(batch_dict) if batch_dict else {}

        for slot in self.slots:
            if slot.output_data_key in result:
                continue
            values = [self._get_sample_values(s, slot) for s in batch]
            data, mask = slot.processor.process_batch(values)
            if data is not None:
                result[slot.output_data_key] = data
            if mask is not None and slot.output_mask_key:
                result[slot.output_mask_key] = mask

        return BatchFeature(result)


class _LegacyMetaProcessorBase(MultimodalMetaProcessor):
    """
    Base class for deprecated task-specific processor wrappers.

    Legacy processors (e.g. Text2TextTranslationProcessor) accept ``tokenizer``
    as a build-time convenience parameter to pass into their TextModalityProcessor
    slots. ProcessorMixin 5.x would misinterpret that parameter name as a required
    typed modality component, so this base class overrides ``get_attributes()`` to
    return [] — the same natural result that MultimodalMetaProcessor itself produces,
    but explicitly enforced so that subclass __init__ signatures do not re-trigger
    the detection.

    All legacy complexity is isolated here; the modern MultimodalMetaProcessor API
    needs no such override.
    """

    @classmethod
    def get_attributes(cls):
        return []

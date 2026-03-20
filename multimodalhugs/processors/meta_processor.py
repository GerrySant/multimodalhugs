import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from multimodalhugs.processors.modality_processor import ModalityProcessor

logger = logging.getLogger(__name__)


@dataclass
class ProcessorSlot:
    """
    Binds a ModalityProcessor to a set of dataset item fields and a set of
    forward() argument names.

    column_map  — maps dataset item field name → processor parameter name.
                  The first key is the *primary* field: its value is replaced
                  with a preprocessed tensor by _transform_get_items_output.
                  All other keys are context-only (e.g. temporal bounds) and
                  are passed to process_sample() but not written back.
                  Default {"signal": "signal"} covers the standard single-field
                  case and requires no explicit configuration.

    output_data_key  — the key under which the data tensor is stored in the
                       BatchFeature returned by MultimodalMetaProcessor.__call__.
    output_mask_key  — the key for the mask tensor (None if no mask is needed).
    """
    processor: ModalityProcessor
    output_data_key: str
    output_mask_key: Optional[str] = None
    column_map: Dict[str, str] = field(
        default_factory=lambda: {"signal": "signal"}
    )

    @property
    def primary_field(self) -> str:
        """The dataset item field whose value is replaced with a tensor."""
        return next(iter(self.column_map))


class MultimodalMetaProcessor(ProcessorMixin):
    """
    Orchestrates a collection of ProcessorSlots to produce a full model batch.

    Replaces all task-specific processors (Pose2TextTranslationProcessor, etc.)
    with a composable, slot-based design.

    encoder_slots       — one slot per encoder input stream
    label_slot          — slot for the target sequence (labels)
    encoder_prompt_slot — optional slot for the encoder language/task token
    decoder_prompt_slot — optional slot for the decoder start token
    tokenizer           — kept for HF ProcessorMixin compatibility

    The MetaProcessor is registered with the dataset via:
        dataset.with_transform(meta_processor._transform_get_items_output)
    and is called by the DataCollator via:
        meta_processor(batch_of_samples)
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"
    name = "multimodal_meta_processor"

    def __init__(
        self,
        encoder_slots: List[ProcessorSlot],
        label_slot: ProcessorSlot,
        encoder_prompt_slot: Optional[ProcessorSlot] = None,
        decoder_prompt_slot: Optional[ProcessorSlot] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.encoder_slots = encoder_slots
        self.label_slot = label_slot
        self.encoder_prompt_slot = encoder_prompt_slot
        self.decoder_prompt_slot = decoder_prompt_slot
        super().__init__(tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sample_values(self, sample: Dict[str, Any], slot: "ProcessorSlot") -> Any:
        """
        Extract and optionally preprocess a value from a sample dict for a slot.

        If the primary field value is already a tensor (pre-processed by
        _transform_get_items_output), return it directly.  Otherwise, call
        process_sample so that __call__ works even without a prior transform.
        """
        primary = slot.primary_field
        value = sample[primary]
        if isinstance(value, torch.Tensor):
            return value
        # Not yet a tensor — run process_sample now
        if len(slot.column_map) > 1:
            values = {param: sample[col] for col, param in slot.column_map.items()}
        else:
            values = value
        return slot.processor.process_sample(values)

    def _all_slots(self) -> List[ProcessorSlot]:
        slots = list(self.encoder_slots)
        if self.encoder_prompt_slot:
            slots.append(self.encoder_prompt_slot)
        if self.decoder_prompt_slot:
            slots.append(self.decoder_prompt_slot)
        slots.append(self.label_slot)
        return slots

    # ------------------------------------------------------------------
    # Dataset-level transform  (called via dataset.with_transform)
    # ------------------------------------------------------------------

    def _transform_get_items_output(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Convert raw dataset item fields to tensors for each slot's primary field.
        Registered with dataset.with_transform() — runs before batching.
        """
        for slot in self._all_slots():
            primary = slot.primary_field
            if primary not in batch:
                continue
            n = len(batch[primary])
            batch[primary] = [
                slot.processor.process_sample(
                    {param: batch[col][i] for col, param in slot.column_map.items()}
                    if len(slot.column_map) > 1
                    else batch[primary][i]
                )
                for i in range(n)
            ]
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

        batch_dict — pre-populated dict (e.g. from DataCollator); keys already
                     present are not overwritten (except for new modality keys).
        """
        result = batch_dict or {}

        # Encoder slots
        for slot in self.encoder_slots:
            data, mask = slot.processor.process_batch(
                [self._get_sample_values(s, slot) for s in batch]
            )
            result[slot.output_data_key] = data
            if slot.output_mask_key is not None and mask is not None:
                result[slot.output_mask_key] = mask

        # Encoder prompt slot
        if self.encoder_prompt_slot:
            slot = self.encoder_prompt_slot
            data, mask = slot.processor.process_batch(
                [self._get_sample_values(s, slot) for s in batch]
            )
            result[slot.output_data_key] = data
            if slot.output_mask_key is not None and mask is not None:
                result[slot.output_mask_key] = mask

        # Decoder prompt slot — skip if already populated (e.g. by DataCollator)
        if self.decoder_prompt_slot:
            slot = self.decoder_prompt_slot
            if slot.output_data_key not in result:
                data, mask = slot.processor.process_batch(
                    [self._get_sample_values(s, slot) for s in batch]
                )
                result[slot.output_data_key] = data
                if slot.output_mask_key is not None and mask is not None:
                    result[slot.output_mask_key] = mask

        # Label slot — receives full sample dicts (needs decoder_prompt + output)
        if self.label_slot:
            slot = self.label_slot
            labels, _ = slot.processor.process_batch(batch)
            if labels is not None:
                result[slot.output_data_key] = labels

        return BatchFeature(result)

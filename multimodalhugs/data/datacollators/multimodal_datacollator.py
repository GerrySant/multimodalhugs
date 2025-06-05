import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class DataCollatorMultimodalSeq2Seq:
    processor: Any
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __init__(
        self,
        processor: Any,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def _obtain_text_related_inputs_outputs(self, samples, return_tensors=None):
        batch = {}
        if return_tensors is None:
            return_tensors = self.return_tensors

        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD

        if any(sample['output'] is None for sample in samples):
            labels = None
        
        else:

            labels = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sample['decoder_prompt']))
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sample['output']))
                + [self.tokenizer.eos_token_id]
                for sample in samples
            ]

        if labels is not None:
            if no_padding:
                batch["labels"] = list(labels)
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                batch["labels"] = [
                    label + [self.label_pad_token_id] * (max_label_length - len(label))
                    if padding_side == "right"
                    else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                    for label in labels
                ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                import numpy as np
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        # else:
        #     batch["labels"] = None

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids
        return batch

    def __call__(
        self, samples: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        batch_dict = self._obtain_text_related_inputs_outputs(samples)

        batch = self.processor(
            batch=samples,
            batch_dict=batch_dict,
            return_tensors="pt"
        )

        return batch
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def create_seq2seq_labels_from_samples(
    samples: List[Dict[str, Union[str, List[int]]]],
    tokenizer: PreTrainedTokenizerBase,
    label_pad_token_id: int = -100,
    padding: Union[bool, str, PaddingStrategy] = True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: str = "pt"
) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:

    """
    Tokenizes and formats decoder labels from a batch of multimodal samples.

    This function processes the 'decoder_prompt' and 'output' fields from each sample by
    tokenizing them, concatenating the resulting token IDs, appending an EOS token,
    and optionally padding them to a common length.

    Args:
        samples (List[Dict[str, Union[str, List[int]]]]): A list of samples, each containing
            the fields 'decoder_prompt' and 'output' (both strings expected).
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for tokenizing text fields and
            converting tokens to IDs.
        label_pad_token_id (int, optional): The ID used to pad sequences. This is typically
            set to -100 so that the loss function ignores padding tokens. Defaults to -100.
        padding (Union[bool, str, PaddingStrategy], optional): Padding strategy to apply. If False,
            no padding is applied. If 'max_length' or True, padding is applied based on `max_length`
            or the longest sequence. Defaults to True.
        max_length (Optional[int], optional): The maximum length to pad/truncate the label sequences to.
            Required if padding is 'max_length'. Defaults to None.
        pad_to_multiple_of (Optional[int], optional): If provided, the sequences will be padded to a multiple
            of this value. Useful for hardware optimization. Defaults to None.
        return_tensors (str, optional): The tensor format to return: 'pt' (PyTorch), 'tf' (TensorFlow), or 'np' (NumPy).
            Defaults to 'pt'.

    Returns:
        Optional[Dict[str, Union[torch.Tensor, Any]]]: A dictionary containing:
            - 'labels': A tensor or array of padded token ID sequences.
        If any sample is missing an 'output', returns None.
    """

    # Check for missing outputs
    if any(sample.get('output') is None for sample in samples):
        return None

    # Build raw label sequences
    labels = []
    for sample in samples:
        # Tokenize decoder prompt and output, append EOS token
        prompt_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(sample['decoder_prompt'])
        )
        output_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(sample['output'])
        )
        combined = prompt_ids + output_ids + [tokenizer.eos_token_id]
        labels.append(combined)

    # Determine if padding is disabled
    no_padding = padding is False or padding == PaddingStrategy.DO_NOT_PAD

    if no_padding:
        final_labels = labels
    else:
        # Determine target max length
        if padding == PaddingStrategy.MAX_LENGTH and max_length:
            max_len = max_length
        else:
            max_len = max(len(seq) for seq in labels)

        # Adjust to pad_to_multiple_of
        if pad_to_multiple_of:
            multiple = pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        # Pad sequences on correct side
        padded = []
        side = tokenizer.padding_side
        for seq in labels:
            pad_len = max_len - len(seq)
            if side == 'right':
                padded_seq = seq + [label_pad_token_id] * pad_len
            else:
                padded_seq = [label_pad_token_id] * pad_len + seq
            padded.append(padded_seq)
        final_labels = padded

    # # Convert lists to tensors based on return format
    if return_tensors == 'pt':
        return {'labels': torch.tensor(final_labels, dtype=torch.int64)}
    elif return_tensors == 'tf':
        import tensorflow as tf
        return {'labels': tf.constant(final_labels, dtype=tf.int64)}
    else:
        import numpy as np
        return {'labels': np.array(final_labels, dtype=np.int64)}


@dataclass
class DataCollatorMultimodalSeq2Seq:
    """
    Data collator for multimodal sequence-to-sequence tasks.

    This collator prepares batches that include text labels (e.g., for decoder outputs),
    optionally pads them, and can use the model to generate decoder input IDs.
    It also delegates the multimodal input construction to a provided processor.

    Args:
        processor (Any): A ProcessorMixin object that processes multimodal and related inputs.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to tokenize and convert labels to IDs.
        model (Optional[Any], optional): The model used to optionally prepare decoder input IDs
            from labels. Defaults to None.
        padding (Union[bool, str, PaddingStrategy], optional): Padding strategy, can be True, False,
            'longest', 'max_length', or PaddingStrategy enum. Defaults to True.
        max_length (Optional[int], optional): If set, truncates/pads sequences to this length. Required if
            padding is set to 'max_length'. Defaults to None.
        pad_to_multiple_of (Optional[int], optional): If set, pad sequences to a multiple of this value.
            Useful for hardware optimization (e.g., XLA or Tensor Cores). Defaults to None.
        label_pad_token_id (int, optional): Token ID used to pad the labels. Ignored by loss functions
            like cross-entropy. Defaults to -100.
        return_tensors (str, optional): The format in which to return the batch ('pt', 'tf', or 'np').
            Defaults to 'pt'.
    """
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
        model: Optional[Any] = None, # The model (optional, used to create decoder input IDs if needed)
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        """
        Initialize the multimodal sequence-to-sequence data collator.

        Args:
            processor (Any): A ProcessorMixin object that processes multimodal inputs and merges them with text fields.
            tokenizer (PreTrainedTokenizerBase, optional): Tokenizer to tokenize text inputs and outputs.
                If None, defaults to `processor.tokenizer`.
            model (Optional[Any], optional): The model used to prepare decoder_input_ids from labels.
            padding (Union[bool, str, PaddingStrategy], optional): Strategy for padding labels.
                Can be True, False, 'longest', 'max_length', or a PaddingStrategy.
            max_length (Optional[int], optional): Max length for padding/truncating labels.
            pad_to_multiple_of (Optional[int], optional): Pads to a multiple of this number, if set.
            label_pad_token_id (int, optional): Token ID used to pad label sequences.
            return_tensors (str, optional): Format of output tensors: 'pt', 'tf', or 'np'. Defaults to 'pt'.
        """
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors


    def _obtain_labels_and_decoder_input_ids(
        self,
        samples: List[Dict[str, Union[List[int], torch.Tensor]]],
        return_tensors: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize, pad, and optionally create decoder inputs from text labels.

        Args:
            samples: List of dicts with 'decoder_prompt' and 'output'.
            return_tensors: Override for return_tensors attribute.

        Returns:
            A dict with:
              - 'labels': Padded token ID sequences.
              - 'decoder_input_ids': Prepared IDs if model supports it.
        """
        rt = return_tensors or self.return_tensors
        batch = create_seq2seq_labels_from_samples(
            samples=samples,
            tokenizer=self.tokenizer,
            label_pad_token_id=self.label_pad_token_id,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=rt
        ) or {}
        # Use model to prepare decoder inputs if available
        if 'labels' in batch and self.model and hasattr(self.model, 'prepare_decoder_input_ids_from_labels'):
            batch['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(
                labels=batch['labels']
            )

        return batch

    def __call__(
        self,
        samples: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of multimodal samples for model input.

        Args:
            samples: Each item contains multimodal data and text fields.

        Returns:
            A dict ready for model.forward(), including inputs and labels.
        """
        # Process text side: tokenization, padding, decoder inputs
        text_batch = self._obtain_labels_and_decoder_input_ids(samples)
        # Delegate full example construction to processor
        full_batch = self.processor(
            batch=samples,
            batch_dict=text_batch,
            return_tensors=self.return_tensors,
        )
        return full_batch
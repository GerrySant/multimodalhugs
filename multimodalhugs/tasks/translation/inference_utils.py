import numpy as np
from typing import Dict, Union, Optional, List, Tuple, Any, Literal

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

from transformers.generation.configuration_utils import GenerationConfig
from multimodalhugs.data import DataCollatorMultimodalSeq2Seq

# -----------------------------
# Functions for directly perform inference
# -----------------------------

ModalityType = Literal[
    "pose2text",
    "video2text",
    "features2text",
    "signwriting2text",
    "image2text",
    "text2text",
]

def get_inference_dataloader(processor, tsv_path: str = "", modality: ModalityType = "pose2text", batch_size: int = 1):
    if modality == "pose2text":
        from multimodalhugs.data.datasets.pose2text import Pose2TextDataset as dataset_class
    elif modality == "video2text":
        from multimodalhugs.data.datasets.video2text import Video2TextDataset as dataset_class
    elif modality == "features2text":
        from multimodalhugs.data.datasets.features2text import Features2TextDataset as dataset_class
    elif modality == "signwriting2text":
        from multimodalhugs.data.datasets.signwriting import SignWritingDataset as dataset_class
    elif modality == "image2text":
        from multimodalhugs.data.datasets.bilingual_image2text import BilingualImage2TextDataset as dataset_class
    elif modality == "text2text":
        from multimodalhugs.data.datasets.bilingual_text2text import BilingualText2TextDataset as dataset_class
    else:
        sys.exit(f"Unknown modality: {modality}")

    dataset = dataset_class(test_metadata_file=tsv_path)
    dataset.download_and_prepare()
    dataset = dataset.as_dataset()["test"]
    data_collator = DataCollatorMultimodalSeq2Seq(processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def logits_to_text(tokenizer, generated_tokens, labels):
    if isinstance(generated_tokens, tuple):
        generated_tokens = pregenerated_tokensds[0]
    generated_tokens = np.where(generated_tokens != -100, generated_tokens, tokenizer.pad_token_id)
    decoded_generated_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    if labels is not None:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return decoded_generated_tokens, decoded_labels
    else:
        return decoded_generated_tokens, None
        
def all_values_equal(tensor: torch.Tensor) -> bool:
    return torch.all(tensor == tensor.view(-1)[0])

def batched_prediction(
    model: nn.Module,
    tokenizer,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
    prepare_inputs_fn: Optional[callable] = None,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Standalone prediction step function that optionally runs .generate().

    Args:
        model: PyTorch model (nn.Module).
        tokenizer: Tokenizer used for padding.
        inputs: Dict of input tensors (must include 'labels' if computing loss).
        prediction_loss_only: Whether to return only the loss.
        predict_with_generate: If True, runs .generate().
        generation_config: GenerationConfig or dict with generation 
        prepare_inputs_fn: Optional function to preprocess inputs (e.g., move to device).
        gen_kwargs: Additional generate kwargs (overrides generation_config).

    Returns:
        (loss (optional), generated_tokens or logits, labels (optional))
    """
    # Prepare inputs
    if prepare_inputs_fn is not None:
        inputs = prepare_inputs_fn(inputs)

    has_labels = "labels" in inputs
    labels = inputs["labels"] if has_labels else None

    # Generation mode
    gen_args = {}
    if isinstance(generation_config, GenerationConfig):
        gen_args = generation_config.to_dict()
    elif isinstance(generation_config, dict):
        gen_args = generation_config.copy()
    if gen_kwargs is not None:
        gen_update({k: v for k, v in gen_kwitems() if v is not None})

    generation_inputs = inputs.copy()

    # Remove decoder inputs if they match the labels to allow generation
    if (
        "labels" in generation_inputs
        and "decoder_input_ids" in generation_inputs
        and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
    ):
        generation_inputs = {
            k: v for k, v in generation_inputs.items()
            if k not in ("decoder_input_ids", "decoder_attention_mask")
        }

    # Case 1: all decoder prompts same length → batch generate
    if "decoder_attention_mask" in generation_inputs and all_values_equal(generation_inputs["decoder_attention_mask"]):
        with torch.no_grad():
            generated_tokens = model.generate(**generation_inputs, **gen_args)

    # Case 2: no decoder prompts → generate from scratch
    elif "decoder_attention_mask" in generation_inputs and generation_inputs["decoder_attention_mask"].numel() == 0:
        generation_inputs.pop("decoder_input_ids", None)
        generation_inputs.pop("decoder_attention_mask", None)
        with torch.no_grad():
            generated_tokens = model.generate(**generation_inputs, **gen_args)

    # Case 3: variable-length prompts → generate sample by sample
    else:
        B = next(iter(generation_inputs.values())).shape[0]
        samples = [{k: v[i:i+1] for k, v in generation_inputs.items()} for i in range(B)]

        generated_tokens = []
        max_len = 0

        for sample in samples:
            with torch.no_grad():
                out = model.generate(**sample, **gen_args)
            max_len = max(max_len, out.shape[1])
            generated_tokens.append(out)

        pad_id = tokenizer.pad_token_id
        for i in range(len(generated_tokens)):
            pad_len = max_len - generated_tokens[i].size(1)
            if pad_len > 0:
                generated_tokens[i] = F.pad(generated_tokens[i], (0, pad_len), value=pad_id)

        generated_tokens = torch.cat(generated_tokens, dim=0)

    if has_labels:
        return (generated_tokens.detach().cpu(), labels.detach().cpu())
    else:
        return (generated_tokens.detach().cpu(), None)

def batched_inference(model, processor, tsv_path, modality, batch_size: int = 1):
    # Logic
    dataloader = get_inference_dataloader(processor=processor, tsv_path=tsv_path, modality=modality, batch_size=batch_size)
    predicted_samples = []
    labels_list = []
    for batch in dataloader:
        batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        with torch.no_grad():
            generated_tokens, labels = batched_prediction(
                model=model,
                tokenizer=processor.tokenizer,
                inputs=batch,
            )
            generated_tokens, labels = logits_to_text(processor.tokenizer, generated_tokens, labels)
            predicted_samples.extend(generated_tokens)
            if labels is not None:
                labels_list.extend(labels)
    return {
        "preds": predicted_samples, 
        "labels": labels_list if len(labels_list) == len(predicted_samples) else []
        }




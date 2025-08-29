# multimodalhugs/multimodalhugs/tasks/translation/inference_utils.py
# -*- coding: utf-8 -*-
"""
Inference utilities for MultimodalHugs translation-style tasks.

What this module does
---------------------
1) Creates a dataloader from a TSV metadata file for several modalities.
2) Runs batched text generation with Hugging Face `model.generate`.
3) Decodes generated token IDs back to strings.
4) Computes a **per-sample perplexity** for each generated prediction.
   - Perplexity is computed *after* generation by re-scoring the chosen
     output tokens with a teacher-forcing forward pass. This makes it work
     robustly for greedy, sampling, and beam search and for both
     encoder-decoder and decoder-only models.

Returned structure (from `batched_inference`)
---------------------------------------------
{
    "preds": [str, ...],              # decoded predictions
    "labels": [str, ...] | [],        # decoded references if they exist
    "perplexities": [float, ...] | [] # one number per prediction (lower is better)
}

Design choices
--------------
- We ask `generate` to return a dict (`return_dict_in_generate=True`) and
  the per-step logits (`output_scores=True`). We then align the generated
  tokens with those scores using the rule: the **last T tokens** of the
  returned sequences correspond to the **T generation steps**, where
  T = len(outputs.scores).
- For encoder-decoder models, we rebuild a decoder prompt consisting of the
  preceding generated tokens (teacher forcing) and pass the original
  encoder inputs again to obtain the logits for the generated segment.
- For decoder-only models, we forward the sequence excluding the last token
  so that `logits[:, -T:, :]` aligns with the T generated tokens.

Notes & caveats
---------------
- If the tokenizer has no PAD token, we fall back to EOS or `0` for masking.
- If `generate` can't return scores (e.g., some special models), perplexities
  will be an empty list.
- The dataloader is built inside a TemporaryDirectory cache. This matches the
  original pattern. If your dataset implementation memory-maps files, consider
  managing cache lifetime externally.

"""

from __future__ import annotations

import sys
import tempfile
from typing import Dict, Union, Optional, List, Tuple, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig

from multimodalhugs.data import DataCollatorMultimodalSeq2Seq


# -----------------------------
# Public types
# -----------------------------

ModalityType = Literal[
    "pose2text",
    "video2text",
    "features2text",
    "signwriting2text",
    "image2text",
    "text2text",
]


# -----------------------------
# Dataloader creation
# -----------------------------

def get_inference_dataloader(
    processor,
    tsv_path: str = "",
    modality: ModalityType = "pose2text",
    batch_size: int = 1,
) -> DataLoader:
    """
    Build an inference DataLoader for the requested modality, given a TSV metadata file.

    Parameters
    ----------
    processor : Any
        The object that bundles the tokenizer and feature processor used by the model.
        It must be compatible with `DataCollatorMultimodalSeq2Seq(processor)`.
    tsv_path : str
        Path to a TSV file with test metadata (dataset-specific schema).
    modality : ModalityType
        Which dataset class to use (determines how rows are parsed/preprocessed).
    batch_size : int
        Batch size for inference (generation is typically memory-bound; start small).

    Returns
    -------
    DataLoader
        A PyTorch DataLoader that yields dictionaries compatible with the model/processor.
    """
    # Dynamically import the dataset class (keeps dependencies minimal per modality)
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

    # Build dataset and collator
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = dataset_class(test_metadata_file=tsv_path, cache_dir=tmp_dir)
        dataset.download_and_prepare()
        dataset = dataset.as_dataset()["test"]
        data_collator = DataCollatorMultimodalSeq2Seq(processor)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )


# -----------------------------
# Small helpers
# -----------------------------

def postprocess_text(preds: List[str], labels: List[str]) -> Tuple[List[str], List[List[str]]]:
    """
    Whitespace-strip predictions and labels.

    Returns
    -------
    (preds, labels_wrapped)
        `labels_wrapped` is a list of lists, which some metrics libraries expect.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def _to_numpy(x: Optional[Union[np.ndarray, torch.Tensor]]) -> Optional[np.ndarray]:
    """Convert a tensor to numpy if needed; pass None through."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type for conversion to numpy: {type(x)}")


def logits_to_text(
    tokenizer,
    generated_tokens: Union[torch.Tensor, np.ndarray, Tuple],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Decode token IDs → strings, masking ignore index (-100) to tokenizer.pad_token_id.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
    generated_tokens : torch.Tensor | np.ndarray | tuple
        Token IDs returned by `model.generate(...)`. Some models return a tuple where
        the first element is the `sequences` tensor.
    labels : torch.Tensor | np.ndarray | None
        Optional ground-truth token IDs (for evaluation); will be decoded if provided.

    Returns
    -------
    (decoded_generated, decoded_labels | None)
    """
    # Some HF models return tuples; extract the sequences
    if isinstance(generated_tokens, tuple):
        generated_tokens = generated_tokens[0]

    # Convert to numpy (works with np.where below)
    generated_tokens = _to_numpy(generated_tokens)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Fall back to EOS or 0 if PAD is not set
        pad_id = getattr(tokenizer, "eos_token_id", 0)

    # Replace ignore index with PAD for clean decoding
    generated_tokens = np.where(generated_tokens != -100, generated_tokens, pad_id)
    decoded_generated_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    if labels is not None:
        labels = _to_numpy(labels)
        labels = np.where(labels != -100, labels, pad_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return decoded_generated_tokens, decoded_labels

    return decoded_generated_tokens, None


def all_values_equal(tensor: torch.Tensor) -> bool:
    """
    True if all entries in a tensor are equal (useful to detect uniform decoder masks).
    """
    if tensor.numel() == 0:
        return True
    return bool(torch.all(tensor == tensor.view(-1)[0]))


# -----------------------------
# Perplexity computation
# -----------------------------

def _compute_perplexities_from_generate(
    model: nn.Module,
    tokenizer,
    outputs,                                   # return dict from model.generate (must have .sequences)
    generation_inputs: Dict[str, Union[torch.Tensor, Any]],
) -> List[float]:
    """
    Compute per-sample perplexity for the generated segment.

    How it works
    ------------
    Let T be the number of generation steps (`T = len(outputs.scores)`).
    The last T tokens of `outputs.sequences` are exactly the generated tokens.
    We re-score those T tokens with a teacher-forcing forward pass:
      - encoder-decoder: pass original encoder inputs + the preceding decoder tokens
      - decoder-only   : forward the entire sequence except last token and slice logits

    We then gather the log-prob of the actually chosen token at each step and
    compute:
        ppl = exp( - mean(log p_token) ) , averaged over non-PAD positions.

    Parameters
    ----------
    model : nn.Module
    tokenizer : PreTrainedTokenizerBase
    outputs : GenerateDecoderOnlyOutput | GenerateEncoderDecoderOutput | ...
        The return value from `model.generate` with `return_dict_in_generate=True`.
    generation_inputs : dict
        The original inputs used for generation (so we can reconstruct encoder inputs).

    Returns
    -------
    List[float]
        Perplexity for each item in the batch. Empty if scores are unavailable.
    """
    if not hasattr(outputs, "sequences"):
        return []

    sequences: torch.Tensor = outputs.sequences  # shape [B, L]
    device = sequences.device

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)

    # Number of steps the generator actually took (one score per generated token)
    n_steps = len(getattr(outputs, "scores", []) or [])
    if n_steps == 0:
        # Some models/configs may not produce scores; skip perplexity.
        return []

    # The generated tokens are the trailing n_steps of sequences
    gen_tokens = sequences[:, -n_steps:].to(device)  # [B, T]

    with torch.no_grad():
        if getattr(model.config, "is_encoder_decoder", False):
            # Teacher-forcing decoder inputs = the preceding tokens for each generated step
            dec_inp = sequences[:, -n_steps-1:-1].to(device)  # [B, T]

            # Filter out decoder-specific fields; keep encoder inputs only.
            enc_kwargs = {
                k: v for k, v in generation_inputs.items()
                if k not in ("labels", "decoder_input_ids", "decoder_attention_mask")
            }
            enc_kwargs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc_kwargs.items()}

            # Forward pass to get logits aligned with generated steps: [B, T, V]
            logits = model(**enc_kwargs, decoder_input_ids=dec_inp).logits
        else:
            # Decoder-only: forward all tokens except the last so that
            # logits[:, -T:, :] aligns with the T generated positions.
            full_inp = sequences[:, :-1].to(device)          # [B, L-1]
            logits = model(input_ids=full_inp).logits[:, -n_steps:, :]  # [B, T, V]

        # Convert logits → log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)                              # [B, T, V]
        chosen_logp = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T]

        # Mask PAD positions (these usually appear after EOS)
        mask = (gen_tokens != pad_id).float()
        token_counts = mask.sum(dim=1).clamp_min(1.0)                          # [B]
        mean_logp = (chosen_logp * mask).sum(dim=1) / token_counts             # [B]
        ppl = torch.exp(-mean_logp).detach().cpu().tolist()                    # [B]
        return ppl


# -----------------------------
# Core prediction/generation
# -----------------------------

def batched_prediction(
    model: nn.Module,
    tokenizer,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
    prepare_inputs_fn: Optional[callable] = None,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    return_perplexity: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[float]]]:
    """
    Perform a single batched prediction step:
      - Optionally preprocess inputs (e.g., move to device)
      - Run `model.generate` (handling different prompt shapes)
      - Optionally compute per-sample perplexity for the generated segment

    Parameters
    ----------
    model : nn.Module
        A Hugging Face-compatible generation model.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used by the model. Needed for PAD/EOS and decoding.
    inputs : dict
        Batch dict compatible with the model (and optionally containing "labels").
    generation_config : GenerationConfig | dict | None
        Default generation parameters. A dict is fine; we `to_dict()` if needed.
    prepare_inputs_fn : callable | None
        A function(inputs) -> inputs, e.g., to move all tensors to model.device.
    gen_kwargs : dict | None
        Additional keyword args for `model.generate` that override `generation_config`.
    return_perplexity : bool
        If True, compute a per-sample perplexity list.

    Returns
    -------
    (generated_tokens, labels | None, perplexities | None)
        generated_tokens : torch.LongTensor with shape [B, L_generated]
        labels           : torch.LongTensor or None
        perplexities     : List[float] or None
    """
    # Optional preprocessing (e.g., device move, precision cast)
    if prepare_inputs_fn is not None:
        inputs = prepare_inputs_fn(inputs)

    has_labels = "labels" in inputs
    labels = inputs["labels"] if has_labels else None

    # Merge generation config / kwargs
    gen_args: Dict[str, Any] = {}
    if isinstance(generation_config, GenerationConfig):
        gen_args = generation_config.to_dict()
    elif isinstance(generation_config, dict):
        gen_args = generation_config.copy()

    if gen_kwargs is not None:
        # Override defaults with explicit kwargs
        gen_args.update({k: v for k, v in gen_kwargs.items() if v is not None})

    # Ensure we get a rich return payload for perplexity
    gen_args.setdefault("return_dict_in_generate", True)
    gen_args.setdefault("output_scores", True)
    gen_args.setdefault("pad_token_id", tokenizer.pad_token_id)

    generation_inputs = inputs.copy()

    # If both `labels` and `decoder_input_ids` are present with same shape, HF may
    # think we want forced decoding. Remove decoder fields to allow free generation.
    if (
        "labels" in generation_inputs
        and "decoder_input_ids" in generation_inputs
        and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
    ):
        generation_inputs = {
            k: v for k, v in generation_inputs.items()
            if k not in ("decoder_input_ids", "decoder_attention_mask")
        }

    # ---- Generation paths -----------------------------------------------------
    # We distinguish 3 cases for efficiency/stability:
    #   1) Uniform decoder_attention_mask (same prompt length for all) → batch generate
    #   2) decoder masks exist but are empty → generate from scratch
    #   3) Variable-length prompts → generate sample-by-sample and concatenate

    use_batch_path = False
    if "decoder_attention_mask" in generation_inputs:
        dam = generation_inputs["decoder_attention_mask"]
        # PyTorch tensors assumed
        if isinstance(dam, torch.Tensor) and dam.numel() > 0 and all_values_equal(dam):
            use_batch_path = True

    if use_batch_path:
        # Case 1: batch generation
        with torch.no_grad():
            outputs = model.generate(**generation_inputs, **gen_args)

        generated_tokens = outputs.sequences.detach().cpu()
        labels_cpu = labels.detach().cpu() if has_labels else None

        perplexities = None
        if return_perplexity:
            perplexities = _compute_perplexities_from_generate(model, tokenizer, outputs, generation_inputs)

        return generated_tokens, labels_cpu, perplexities

    # Case 2: generate from scratch (no decoder prompts)
    if (
        "decoder_attention_mask" in generation_inputs
        and isinstance(generation_inputs["decoder_attention_mask"], torch.Tensor)
        and generation_inputs["decoder_attention_mask"].numel() == 0
    ):
        generation_inputs.pop("decoder_input_ids", None)
        generation_inputs.pop("decoder_attention_mask", None)
        with torch.no_grad():
            outputs = model.generate(**generation_inputs, **gen_args)

        generated_tokens = outputs.sequences.detach().cpu()
        labels_cpu = labels.detach().cpu() if has_labels else None

        perplexities = None
        if return_perplexity:
            perplexities = _compute_perplexities_from_generate(model, tokenizer, outputs, generation_inputs)

        return generated_tokens, labels_cpu, perplexities

    # Case 3: variable-length prompts → generate per-sample then pad+concat
    B = next(iter(generation_inputs.values())).shape[0]  # batch size
    samples = [
        {
            k: (v[i:i+1] if isinstance(v, torch.Tensor) else v)
            for k, v in generation_inputs.items()
        }
        for i in range(B)
    ]

    seqs: List[torch.Tensor] = []
    perplexities: List[float] = []
    max_len = 0
    pad_id = tokenizer.pad_token_id

    for sample in samples:
        with torch.no_grad():
            out_i = model.generate(**sample, **gen_args)  # returns dict-like output
        seqs.append(out_i.sequences)
        max_len = max(max_len, out_i.sequences.shape[1])

        if return_perplexity:
            ppl_i = _compute_perplexities_from_generate(model, tokenizer, out_i, sample)
            perplexities.extend(ppl_i)

    # Right-pad each sequence to the same length, then concatenate on batch dim
    for i in range(len(seqs)):
        pad_len = max_len - seqs[i].size(1)
        if pad_len > 0:
            seqs[i] = F.pad(seqs[i], (0, pad_len), value=pad_id)

    generated_tokens = torch.cat(seqs, dim=0).detach().cpu()
    labels_cpu = labels.detach().cpu() if has_labels else None

    return generated_tokens, labels_cpu, (perplexities if return_perplexity else None)


# -----------------------------
# End-to-end inference loop
# -----------------------------

def batched_inference(
    model: nn.Module,
    processor,
    tsv_path: str,
    modality: ModalityType,
    batch_size: int = 1,
) -> Dict[str, List[Any]]:
    """
    End-to-end convenience function:
        1) Build dataloader
        2) Generate predictions (batched)
        3) Decode to strings
        4) Return predictions, labels (if available), and per-sample perplexities.

    Parameters
    ----------
    model : nn.Module
        A model with `.generate(...)` (e.g., T5, BART, LLaMA, mT5, etc.).
        `model.device` must be set correctly (e.g., move with `model.to(device)`).
    processor : Any
        Object containing the tokenizer and feature processor (e.g., `AutoProcessor`).
        Must be compatible with `DataCollatorMultimodalSeq2Seq`.
    tsv_path : str
        Path to TSV with test data.
    modality : ModalityType
        Modality key selecting the dataset class.
    batch_size : int
        Inference batch size.

    Returns
    -------
    dict
        {
          "preds": [...],
          "labels": [...] or [],
          "perplexities": [...] or [],
        }
    """
    dataloader = get_inference_dataloader(
        processor=processor,
        tsv_path=tsv_path,
        modality=modality,
        batch_size=batch_size,
    )

    predicted_samples: List[str] = []
    labels_list: List[str] = []
    perplexities_all: List[float] = []

    for batch in tqdm(dataloader):
        # Ensure tensors are on the model's device (non-tensors are passed through)
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}

        with torch.no_grad():
            generated_tokens, labels, perplexities = batched_prediction(
                model=model,
                tokenizer=processor.tokenizer,
                inputs=batch,
                return_perplexity=True,  # flip to False if you don't need perplexities
            )

        # Decode to text. Accepts torch tensors directly.
        decoded_preds, decoded_labels = logits_to_text(
            processor.tokenizer,
            generated_tokens,
            labels,
        )

        predicted_samples.extend(decoded_preds)

        if decoded_labels is not None:
            labels_list.extend(decoded_labels)

        if perplexities is not None:
            perplexities_all.extend(perplexities)

    return {
        "preds": predicted_samples,
        "labels": labels_list if len(labels_list) == len(predicted_samples) else [],
        "perplexities": perplexities_all if len(perplexities_all) == len(predicted_samples) else [],
    }


# -----------------------------
# (Optional) Minimal example
# -----------------------------
# Run like:
#   MMH_MODEL=/path/to/model-or-hub-id \
#   MMH_PROCESSOR=/path/to/processor-or-hub-id \
#   MMH_TSV=/path/to/inference.tsv \
#   MMH_MODALITY=features2text \
#   python inference_utils.py
#
# Notes:
# - MODALITY must match your dataset ("features2text", "pose2text", "text2text", ...).
# - MODEL/PROCESSOR can be local paths or HF hub IDs.
#
# if __name__ == "__main__":
#     import os
#     import torch
#     import multimodalhugs.models  # registers custom models
#     from transformers import AutoModelForSeq2SeqLM, AutoProcessor
#
#     MODEL_ID = os.getenv("MMH_MODEL", "path/to/model-or-hub-id")
#     PROCESSOR_ID = os.getenv("MMH_PROCESSOR", "path/to/processor-or-hub-id")
#     TSV_PATH = os.getenv("MMH_TSV", "path/to/inference.tsv")
#     MODALITY = os.getenv("MMH_MODALITY", "features2text")
#
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
#     processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
#
#     result = batched_inference(
#         model=model,
#         processor=processor,
#         tsv_path=TSV_PATH,
#         modality=MODALITY,
#         batch_size=8,
#     )
#
#     print("Sample preds (first 5):", result.get("preds", [])[:5])
#     print("Sample perplexities (first 5):", result.get("perplexities", [])[:5])
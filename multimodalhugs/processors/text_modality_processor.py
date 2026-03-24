from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from multimodalhugs.processors.modality_processor import ModalityProcessor


class TextModalityProcessor(ModalityProcessor):
    """
    Tokenizes and processes text for different roles in the pipeline.

    role="prompt" / "encoder"
        process_batch receives a list of strings.
        Returns (token_ids [B, L], attention_mask [B, L]).

    role="label"
        process_batch receives a list of sample dicts, each with
        "decoder_prompt" and "output" keys.
        Concatenates decoder_prompt + output + EOS, pads with -100.
        Returns (labels [B, L], None).
        If any sample has output=None, returns (None, None).

    tokenizer      — a pre-built tokenizer object.
    tokenizer_path — path or HF Hub ID to load the tokenizer from.
                     Used when constructing from a declarative YAML config.
                     Ignored if tokenizer is provided directly.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        tokenizer_path: Optional[str] = None,
        role: str = "prompt",
    ):
        assert role in ("prompt", "encoder", "label"), (
            f"role must be 'prompt', 'encoder', or 'label', got '{role}'"
        )
        if tokenizer is None and tokenizer_path is not None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.role = role

    # ------------------------------------------------------------------
    # ModalityProcessor interface
    # ------------------------------------------------------------------

    def process_sample(
        self,
        values: Union[Any, Dict[str, Any]],
        **kwargs,
    ) -> Any:
        """Text needs no per-sample preprocessing — pass through as-is."""
        return values

    def process_batch(
        self,
        samples: List[Any],
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.role in ("prompt", "encoder"):
            return self._process_prompt_batch(samples)
        else:
            return self._process_label_batch(samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_prompt_batch(
        self,
        texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def _process_label_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> Tuple[Optional[torch.Tensor], None]:
        if any(s.get("output") is None for s in samples):
            return None, None

        labels = []
        for sample in samples:
            prompt_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sample["decoder_prompt"])
            )
            output_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sample["output"])
            )
            labels.append(prompt_ids + output_ids + [self.tokenizer.eos_token_id])

        max_len = max(len(seq) for seq in labels)
        pad_id = -100
        side = self.tokenizer.padding_side
        padded = []
        for seq in labels:
            pad_len = max_len - len(seq)
            if side == "right":
                padded.append(seq + [pad_id] * pad_len)
            else:
                padded.append([pad_id] * pad_len + seq)

        return torch.tensor(padded, dtype=torch.int64), None

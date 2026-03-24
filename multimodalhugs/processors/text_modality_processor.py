from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from multimodalhugs.processors.modality_processor import ModalityProcessor


class TextModalityProcessor(ModalityProcessor):
    """
    Tokenizes and processes text for different roles in the pipeline.

    role="input"
        process_batch receives a list of strings.
        Returns (token_ids [B, L], attention_mask [B, L]).

    role="target"
        process_batch receives a list of sample dicts, each with
        "target_prefix" and "target" keys.
        Concatenates target_prefix + target + EOS, pads with -100.
        Returns (labels [B, L], None).
        If any sample has target=None, returns (None, None).

    tokenizer           — a pre-built tokenizer object.
    tokenizer_path      — path or HF Hub ID to load the tokenizer from.
                          Used when constructing from a declarative YAML config.
                          Ignored if tokenizer is provided directly.
    new_vocabulary      — path to a vocabulary file whose tokens are added as
                          special tokens to the tokenizer. After extension,
                          self.new_tokens holds the list of added tokens and
                          self.pretrained_tokenizer holds the unextended copy.
                          TODO: new_tokens / pretrained_tokenizer are exposed here
                          so that setup scripts can derive them from the built
                          processor instead of calling load_tokenizers separately.
                          Once model construction is refactored to read new_tokens
                          from the processor, the load_tokenizers call in the
                          setup files' declarative branch can be removed entirely.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        tokenizer_path: Optional[str] = None,
        new_vocabulary: Optional[str] = None,
        role: str = "prompt",
    ):
        assert role in ("input", "target"), (
            f"role must be 'input' or 'target', got '{role}'"
        )
        if tokenizer is None and tokenizer_path is not None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.pretrained_tokenizer = tokenizer
        self.new_tokens: List[str] = []

        if new_vocabulary is not None and tokenizer is not None:
            # TODO: this extension logic duplicates setup_utils.load_tokenizers.
            # It is intentionally kept here as a temporary bridge: the processor
            # now owns vocabulary extension so the declarative YAML path does not
            # need to call load_tokenizers. The duplication will be resolved when
            # the model-construction step is refactored to derive new_tokens from
            # the processor directly, at which point load_tokenizers in setup files
            # can be removed entirely.
            from multimodalhugs.utils.tokenizer_utils import extend_tokenizer
            base_path = tokenizer_path or tokenizer.name_or_path
            tokenizer, self.new_tokens = extend_tokenizer(base_path, new_vocabulary)

        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.new_vocabulary = new_vocabulary
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
        if self.role == "input":
            return self._process_prompt_batch(samples)
        elif self.role == "target":
            return self._process_label_batch(samples)
        else:
            raise ValueError(f"Unknown role '{self.role}'. Must be 'input' or 'target'.")

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
        if any(s.get("target") is None for s in samples):
            return None, None

        labels = []
        for sample in samples:
            prompt_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sample["target_prefix"])
            )
            output_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sample["target"])
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

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch

from multimodalhugs.processors.modality_processor import ModalityProcessor, ProcessBatchOutput


class TextRole(str, Enum):
    """Processing role for TextModalityProcessor.

    INPUT  — tokenise plain strings; returns (ids, attention_mask).
    TARGET — concatenate target_prefix + target + EOS; pad with -100;
             returns (ids, None).
    """
    INPUT  = "input"
    TARGET = "target"


class TextModalityProcessor(ModalityProcessor):
    """
    Tokenizes and processes text for different roles in the pipeline.

    TextRole.INPUT
        process_batch receives a list of strings.
        Returns (token_ids [B, L], attention_mask [B, L]).

    TextRole.TARGET
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
        role: Union[TextRole, str] = TextRole.INPUT,
    ):
        """
        Args:
            tokenizer: A pre-built HuggingFace tokenizer instance. When
                provided, ``tokenizer_path`` is ignored for loading but is
                still stored for serialisation. Default: None.
            tokenizer_path: HuggingFace model ID or local path from which the
                tokenizer is loaded via ``AutoTokenizer.from_pretrained`` when
                ``tokenizer`` is None. Default: None.
            new_vocabulary: Path to a vocabulary file (one token per line) or
                a comma-separated string of tokens to add as special tokens to
                the tokenizer. After extension, the added tokens are available
                as ``self.new_tokens`` and the original unextended tokenizer is
                preserved as ``self.pretrained_tokenizer``. Default: None.
            role: Processing role — either ``TextRole.INPUT`` (tokenise input
                strings; returns ids + attention mask) or ``TextRole.TARGET``
                (build labels from target_prefix + target + EOS, padded with
                -100; returns ids only). Accepts the string values ``"input"``
                and ``"target"`` as well as the enum. Default: TextRole.INPUT.
        """
        if isinstance(role, str):
            valid = [r.value for r in TextRole]
            if role not in valid:
                raise ValueError(
                    f"Invalid role '{role}'. Must be one of: {valid}. "
                    f"Check the 'role' argument passed to {self.__class__.__name__}."
                )
            role = TextRole(role)
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
            # can be removed entirely. Tracked in issue #74.
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
    ) -> ProcessBatchOutput:
        if self.role == TextRole.INPUT:
            return self._process_prompt_batch(samples)
        elif self.role == TextRole.TARGET:
            return self._process_label_batch(samples)
        else:
            raise ValueError(f"Unknown role '{self.role}'. Must be TextRole.INPUT or TextRole.TARGET.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_prompt_batch(
        self,
        texts: List[str],
    ) -> ProcessBatchOutput:
        tokenized = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        return ProcessBatchOutput(
            data=tokenized["input_ids"],
            mask=tokenized["attention_mask"],
        )

    def _process_label_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> ProcessBatchOutput:
        # TODO: label construction is currently hardcoded as
        # target_prefix + target + EOS, padded with -100. Customizable label
        # creation strategies (e.g. target-only without prefix, different
        # special tokens, or span masking for MLM) are not yet supported.
        # When needed, this could be delegated to a configurable label_builder
        # callable passed at construction time.
        if any(s.get("target") is None for s in samples):
            return ProcessBatchOutput(data=None, mask=None)

        labels = []
        for sample in samples:
            for key in ("target_prefix", "target"):
                if key not in sample:
                    raise KeyError(
                        f"Sample is missing required key '{key}'. "
                        f"Available keys: {list(sample.keys())}. "
                        "Check that the column_map for this slot maps the correct "
                        "dataset columns to 'target_prefix' and 'target'."
                    )
            prompt_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sample["target_prefix"])
            )
            output_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sample["target"])
            )
            labels.append(prompt_ids + output_ids + [self.tokenizer.eos_token_id])

        max_len = max(len(seq) for seq in labels)
        # -100 is CrossEntropyLoss's default ignore_index — the same convention
        # used throughout HuggingFace Transformers for seq2seq label padding.
        pad_id = -100
        side = self.tokenizer.padding_side
        padded = []
        for seq in labels:
            pad_len = max_len - len(seq)
            if side == "right":
                padded.append(seq + [pad_id] * pad_len)
            else:
                padded.append([pad_id] * pad_len + seq)

        return ProcessBatchOutput(data=torch.tensor(padded, dtype=torch.int64), mask=None)

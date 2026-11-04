"""Tests for add_new_special_tokens_from_vocab_file in utils/tokenizer_utils.py."""

import pytest

from multimodalhugs.utils.tokenizer_utils import add_new_special_tokens_from_vocab_file


class _FakeLangTokenizer:
    """Minimal stub that reproduces M2M100Tokenizer's _extra_special_tokens behaviour.

    M2M100Tokenizer stores its 100 language codes in _extra_special_tokens.
    Calling add_special_tokens({'extra_special_tokens': [...]}) *replaces* that
    list entirely — the same behaviour this stub replicates.
    """

    def __init__(self, lang_codes):
        self._extra_special_tokens = list(lang_codes)
        self._vocab = {}

    def get_vocab(self):
        return self._vocab

    def add_special_tokens(self, d):
        if "extra_special_tokens" in d:
            self._extra_special_tokens = list(d["extra_special_tokens"])


class TestAddNewSpecialTokensPreservation:
    """_extra_special_tokens (M2M100 language codes) must survive add_new_special_tokens_from_vocab_file."""

    def test_language_codes_preserved_when_adding_new_token(self, tmp_path):
        """
        Regression test for commit b59666b: calling add_special_tokens with
        {'extra_special_tokens': [new_token]} replaced _extra_special_tokens entirely,
        wiping all M2M100 language codes and breaking skip_special_tokens=True in
        batch_decode — causing inflated BLEU/chrF scores when language tokens appeared
        in every decoded string.
        """
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("__asl__\n")

        lang_codes = ["__en__", "__fr__", "__de__", "__zh__"]
        tokenizer = _FakeLangTokenizer(lang_codes)

        _, added = add_new_special_tokens_from_vocab_file(tokenizer, str(vocab_file))

        assert added == ["__asl__"]
        for code in lang_codes:
            assert code in tokenizer._extra_special_tokens, (
                f"{code} was lost from _extra_special_tokens — language token wipe bug is present"
            )
        assert "__asl__" in tokenizer._extra_special_tokens
        assert len(tokenizer._extra_special_tokens) == len(lang_codes) + 1

    def test_no_existing_extra_tokens_adds_normally(self, tmp_path):
        """Tokenizers with no pre-existing _extra_special_tokens work unchanged."""
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("__asl__\n")

        tokenizer = _FakeLangTokenizer([])

        _, added = add_new_special_tokens_from_vocab_file(tokenizer, str(vocab_file))

        assert added == ["__asl__"]
        assert tokenizer._extra_special_tokens == ["__asl__"]

    def test_no_duplication_when_token_already_in_vocab(self, tmp_path):
        """Tokens already in the tokenizer vocabulary are skipped, not duplicated."""
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("__en__\n")

        lang_codes = ["__en__", "__fr__"]
        tokenizer = _FakeLangTokenizer(lang_codes)
        tokenizer._vocab = {"__en__": 1}

        _, added = add_new_special_tokens_from_vocab_file(tokenizer, str(vocab_file))

        assert added == []
        assert tokenizer._extra_special_tokens == lang_codes

    def test_comma_separated_vocab_string(self):
        """Comma-separated token string adds multiple tokens while preserving existing extras."""
        lang_codes = ["__en__", "__fr__"]
        tokenizer = _FakeLangTokenizer(lang_codes)

        _, added = add_new_special_tokens_from_vocab_file(tokenizer, "__asl__, __bsl__")

        assert set(added) == {"__asl__", "__bsl__"}
        for code in lang_codes:
            assert code in tokenizer._extra_special_tokens
        assert "__asl__" in tokenizer._extra_special_tokens
        assert "__bsl__" in tokenizer._extra_special_tokens

    def test_token_objects_in_extra_are_stringified(self, tmp_path):
        """Non-string entries in _extra_special_tokens are converted to str before merging."""
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("__asl__\n")

        tokenizer = _FakeLangTokenizer([])
        # Simulate token objects (e.g. AddedToken) stored in _extra_special_tokens
        tokenizer._extra_special_tokens = [type("AddedToken", (), {"__str__": lambda self: "__en__"})()]

        _, added = add_new_special_tokens_from_vocab_file(tokenizer, str(vocab_file))

        assert "__asl__" in tokenizer._extra_special_tokens
        assert any(str(t) == "__en__" for t in tokenizer._extra_special_tokens)

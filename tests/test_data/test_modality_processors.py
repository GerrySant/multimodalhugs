"""
Tests for individual ModalityProcessor classes.

These tests are written before the implementation as a specification (TDD).
They define the expected behaviour of each concrete processor.

Expected module layout (to be created):
    multimodalhugs/processors/modality_processor.py        — ModalityProcessor base
    multimodalhugs/processors/pose_modality_processor.py   — PoseModalityProcessor
    multimodalhugs/processors/video_modality_processor.py  — VideoModalityProcessor
    multimodalhugs/processors/text_modality_processor.py   — TextModalityProcessor

Key design decisions reflected in these tests
----------------------------------------------
* process_sample(raw_value)
    Dataset-level transform (used with dataset.with_transform).
    Receives a single raw value (file path, string, …).
    Returns a pre-loaded value (usually a torch.Tensor).
    Default base-class implementation is a no-op (returns input unchanged).

* process_batch(data: List[Any]) -> Tuple[Tensor, Optional[Tensor]]
    Collator-level transform.  Receives a list of pre-loaded values and
    returns (data_tensor, mask_tensor) after padding to a common length.
    Must be implemented by every concrete subclass.

* TextModalityProcessor role parameter
    role="prompt"  – tokenise plain strings; returns (ids, attention_mask)
    role="encoder" – same as "prompt" (alias for clarity)
    role="label"   – receives List[Dict] with keys "decoder_prompt" and
                     "output"; concatenates them, appends EOS, pads with -100
"""

import pytest
import torch

from multimodalhugs.processors.modality_processor import ModalityProcessor
from multimodalhugs.processors.pose_modality_processor import PoseModalityProcessor
from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor


# ---------------------------------------------------------------------------
# ModalityProcessor base
# ---------------------------------------------------------------------------

class TestModalityProcessorBase:

    def test_process_sample_default_is_noop(self):
        """The base class process_sample should return the input unchanged."""
        class ConcreteProcessor(ModalityProcessor):
            def process_batch(self, data):
                raise NotImplementedError

        proc = ConcreteProcessor()
        raw = "some/path.pose"
        assert proc.process_sample(raw) == raw

    def test_process_batch_must_be_implemented(self):
        """Calling process_batch on a subclass that omits it must raise NotImplementedError."""
        class IncompleteProcessor(ModalityProcessor):
            pass

        with pytest.raises(TypeError):
            # Cannot even instantiate an abstract class if we mark it abstract;
            # but if it is not formally abstract, it should raise on call.
            proc = IncompleteProcessor()
            proc.process_batch([torch.zeros(5)])


# ---------------------------------------------------------------------------
# PoseModalityProcessor
# ---------------------------------------------------------------------------

class TestPoseModalityProcessorProcessSample:

    def test_reads_pose_file(self, dummy_pose_file):
        proc = PoseModalityProcessor(reduce_holistic_poses=True)
        tensor = proc.process_sample(dummy_pose_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 2          # (frames, features)
        assert tensor.shape[0] > 0      # has frames

    def test_tensor_passthrough(self):
        """If the input is already a tensor it should be returned unchanged."""
        proc = PoseModalityProcessor()
        t = torch.randn(10, 64)
        result = proc.process_sample(t)
        assert torch.equal(result, t)

    def test_skip_frames_stride_reduces_temporal_dim(self, dummy_pose_file):
        proc_full = PoseModalityProcessor(reduce_holistic_poses=True)
        proc_skip = PoseModalityProcessor(reduce_holistic_poses=True, skip_frames_stride=2)
        full = proc_full.process_sample(dummy_pose_file)
        skipped = proc_skip.process_sample(dummy_pose_file)
        assert skipped.shape[0] < full.shape[0]
        assert skipped.shape[0] <= (full.shape[0] + 1) // 2 + 1

    def test_reduce_holistic_false_has_more_features(self, dummy_pose_file):
        proc_reduced = PoseModalityProcessor(reduce_holistic_poses=True)
        proc_full    = PoseModalityProcessor(reduce_holistic_poses=False)
        reduced = proc_reduced.process_sample(dummy_pose_file)
        full    = proc_full.process_sample(dummy_pose_file)
        assert full.shape[1] > reduced.shape[1]


class TestPoseModalityProcessorProcessBatch:

    def test_returns_tuple_of_two_tensors(self, dummy_pose_file):
        proc = PoseModalityProcessor(reduce_holistic_poses=True)
        t = proc.process_sample(dummy_pose_file)
        data, mask = proc.process_batch([t])
        assert isinstance(data, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_data_is_3d(self, dummy_pose_file):
        """Output data tensor should be (batch, frames, features)."""
        proc = PoseModalityProcessor(reduce_holistic_poses=True)
        t = proc.process_sample(dummy_pose_file)
        data, _ = proc.process_batch([t, t])
        assert data.ndim == 3
        assert data.shape[0] == 2

    def test_mask_is_2d(self, dummy_pose_file):
        """Output mask tensor should be (batch, frames)."""
        proc = PoseModalityProcessor(reduce_holistic_poses=True)
        t = proc.process_sample(dummy_pose_file)
        _, mask = proc.process_batch([t, t])
        assert mask.ndim == 2
        assert mask.shape[0] == 2

    def test_variable_length_sequences_are_padded(self, dummy_pose_file):
        """Tensors of different lengths must be padded to the same sequence length."""
        proc = PoseModalityProcessor(reduce_holistic_poses=True)
        t = proc.process_sample(dummy_pose_file)
        short = t[:3]   # 3 frames
        long  = t       # full length

        data, mask = proc.process_batch([short, long])
        # Both rows must have the same length
        assert data.shape[1] == long.shape[0]
        # Short sequence should have trailing zeros (or padding)
        assert mask[0].sum() < mask[1].sum()

    def test_mask_is_binary(self, dummy_pose_file):
        proc = PoseModalityProcessor(reduce_holistic_poses=True)
        t = proc.process_sample(dummy_pose_file)
        _, mask = proc.process_batch([t])
        unique = mask.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_data_and_mask_seq_dim_match(self, dummy_pose_file):
        proc = PoseModalityProcessor(reduce_holistic_poses=True)
        t = proc.process_sample(dummy_pose_file)
        data, mask = proc.process_batch([t])
        assert data.shape[1] == mask.shape[1]


# ---------------------------------------------------------------------------
# VideoModalityProcessor
# ---------------------------------------------------------------------------

class TestVideoModalityProcessorProcessSample:

    def test_reads_video_file(self, dummy_video_file):
        proc = VideoModalityProcessor()
        tensor = proc.process_sample(dummy_video_file)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim >= 3     # at minimum (T, H, W) or (T, C, H, W)
        assert tensor.shape[0] > 0

    def test_tensor_passthrough(self):
        proc = VideoModalityProcessor()
        t = torch.randn(5, 3, 64, 64)
        result = proc.process_sample(t)
        assert torch.equal(result, t)


class TestVideoModalityProcessorProcessBatch:

    def test_returns_tuple_of_two_tensors(self, dummy_video_file):
        proc = VideoModalityProcessor()
        t = proc.process_sample(dummy_video_file)
        data, mask = proc.process_batch([t])
        assert isinstance(data, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_data_batch_dim_matches_input(self, dummy_video_file):
        proc = VideoModalityProcessor()
        t = proc.process_sample(dummy_video_file)
        data, mask = proc.process_batch([t, t])
        assert data.shape[0] == 2
        assert mask.shape[0] == 2

    def test_variable_length_sequences_are_padded(self, dummy_video_file):
        proc = VideoModalityProcessor()
        t = proc.process_sample(dummy_video_file)
        short = t[:2]
        long  = t

        data, mask = proc.process_batch([short, long])
        assert data.shape[1] == long.shape[0]
        assert mask[0].sum() < mask[1].sum()


# ---------------------------------------------------------------------------
# TextModalityProcessor — prompt / encoder role
# ---------------------------------------------------------------------------

class TestTextModalityProcessorPromptRole:

    def test_process_batch_returns_tuple(self, tokenizer):
        proc = TextModalityProcessor(tokenizer=tokenizer, role="prompt")
        ids, mask = proc.process_batch(["translate:", "en:"])
        assert isinstance(ids, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_process_batch_batch_dim(self, tokenizer):
        texts = ["translate:", "hello world"]
        proc = TextModalityProcessor(tokenizer=tokenizer, role="prompt")
        ids, mask = proc.process_batch(texts)
        assert ids.shape[0] == len(texts)
        assert mask.shape[0] == len(texts)

    def test_process_batch_ids_and_mask_same_shape(self, tokenizer):
        proc = TextModalityProcessor(tokenizer=tokenizer, role="prompt")
        ids, mask = proc.process_batch(["hello", "hello world how are you"])
        assert ids.shape == mask.shape

    def test_process_batch_pads_variable_length(self, tokenizer):
        proc = TextModalityProcessor(tokenizer=tokenizer, role="prompt")
        ids, mask = proc.process_batch(["hi", "a much longer sentence with more tokens"])
        # Shorter sequence should contain padding (mask value 0)
        assert (mask[0] == 0).any()
        # Longer sequence should be fully attended to
        assert mask[1].all()

    def test_encoder_role_behaves_like_prompt(self, tokenizer):
        texts = ["translate:", "hello world"]
        proc_prompt  = TextModalityProcessor(tokenizer=tokenizer, role="prompt")
        proc_encoder = TextModalityProcessor(tokenizer=tokenizer, role="encoder")
        ids_p, mask_p = proc_prompt.process_batch(texts)
        ids_e, mask_e = proc_encoder.process_batch(texts)
        assert torch.equal(ids_p, ids_e)
        assert torch.equal(mask_p, mask_e)


# ---------------------------------------------------------------------------
# TextModalityProcessor — label role
# ---------------------------------------------------------------------------

class TestTextModalityProcessorLabelRole:
    """
    process_batch for role="label" receives a List[Dict] where each dict
    contains at least "decoder_prompt" and "output" keys.  It mirrors the
    logic currently in create_seq2seq_labels_from_samples:
      label = tokenize(decoder_prompt) + tokenize(output) + [eos_id]
    with -100 padding.
    """

    def _make_samples(self, decoder_prompts, outputs):
        return [
            {"decoder_prompt": dp, "output": out}
            for dp, out in zip(decoder_prompts, outputs)
        ]

    def test_returns_tensor(self, tokenizer):
        proc = TextModalityProcessor(tokenizer=tokenizer, role="label")
        samples = self._make_samples(["de:"], ["Hallo"])
        labels, mask = proc.process_batch(samples)
        assert isinstance(labels, torch.Tensor)

    def test_mask_is_none_for_labels(self, tokenizer):
        """Labels do not need an attention mask — mask should be None."""
        proc = TextModalityProcessor(tokenizer=tokenizer, role="label")
        samples = self._make_samples(["de:"], ["Hallo"])
        _, mask = proc.process_batch(samples)
        assert mask is None

    def test_batch_dim(self, tokenizer):
        proc = TextModalityProcessor(tokenizer=tokenizer, role="label")
        samples = self._make_samples(["de:", "de:"], ["Hallo", "Welt"])
        labels, _ = proc.process_batch(samples)
        assert labels.shape[0] == 2

    def test_eos_token_is_last_real_token(self, tokenizer):
        """Last non-padding token in each sequence must be EOS."""
        proc = TextModalityProcessor(tokenizer=tokenizer, role="label")
        samples = self._make_samples([""], ["Hello"])
        labels, _ = proc.process_batch(samples)
        # Find last non-(-100) token
        row = labels[0]
        real_tokens = row[row != -100]
        assert real_tokens[-1].item() == tokenizer.eos_token_id

    def test_decoder_prompt_appears_before_output(self, tokenizer):
        """The decoder_prompt tokens must appear at the start of the label sequence."""
        proc = TextModalityProcessor(tokenizer=tokenizer, role="label")
        prompt = "de:"
        output = "Hello"
        samples = self._make_samples([prompt], [output])
        labels, _ = proc.process_batch(samples)

        # Tokenise prompt and output separately to get expected ids
        prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
        real_tokens = labels[0][labels[0] != -100].tolist()
        # Labels should start with the prompt token ids
        assert real_tokens[: len(prompt_ids)] == prompt_ids

    def test_shorter_sequence_padded_with_minus_100(self, tokenizer):
        """When sequences differ in length, shorter ones are padded with -100."""
        proc = TextModalityProcessor(tokenizer=tokenizer, role="label")
        samples = self._make_samples(
            ["", ""],
            ["Hi", "A much longer output sentence with many more tokens"],
        )
        labels, _ = proc.process_batch(samples)
        assert labels.shape[0] == 2
        # Both rows same length after padding
        assert labels.shape[1] == labels.shape[1]
        # Shorter row has -100 padding
        assert (labels[0] == -100).any()

    def test_missing_output_returns_none(self, tokenizer):
        """If any sample has output=None, process_batch should return (None, None)."""
        proc = TextModalityProcessor(tokenizer=tokenizer, role="label")
        samples = [
            {"decoder_prompt": "de:", "output": None},
            {"decoder_prompt": "de:", "output": "Hallo"},
        ]
        result, _ = proc.process_batch(samples)
        assert result is None

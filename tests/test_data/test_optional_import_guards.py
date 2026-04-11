"""
Tests that each modality processor and dataset raises a clear ImportError
(mentioning the missing package name) when its optional dependency is absent.

The module-level availability flags (_*_AVAILABLE) are patched via monkeypatch
rather than manipulating sys.modules, because packages such as av, cv2, and
torchvision are also used internally by transformers — blocking them at the
import level would crash the entire package init chain before reaching the code
under test.  Patching the flags directly tests the exact guard logic added to
each __init__ method, which is the behaviour this fix introduces.
"""
import pytest

import multimodalhugs.data.datasets.pose2text as pose2text_mod
import multimodalhugs.data.datasets.signwriting as sw_dataset_mod
import multimodalhugs.data.datasets.video2text as video2text_mod
import multimodalhugs.processors.image_modality_processor as img_mod
import multimodalhugs.processors.pose_modality_processor as pose_mod
import multimodalhugs.processors.signwriting_modality_processor as sw_mod
import multimodalhugs.processors.video_modality_processor as video_mod


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------

def test_pose_processor_raises_without_pose_format(monkeypatch):
    monkeypatch.setattr(pose_mod, "_POSE_FORMAT_AVAILABLE", False)
    with pytest.raises(ImportError, match="pose-format"):
        pose_mod.PoseModalityProcessor()


def test_signwriting_processor_raises_without_signwriting(monkeypatch):
    monkeypatch.setattr(sw_mod, "_SIGNWRITING_AVAILABLE", False)
    with pytest.raises(ImportError, match="signwriting"):
        sw_mod.SignwritingModalityProcessor(custom_preprocessor_path="dummy")


def test_video_processor_raises_without_cv2_when_custom_preprocessor(monkeypatch):
    monkeypatch.setattr(video_mod, "_CV2_AVAILABLE", False)
    with pytest.raises(ImportError, match="opencv-python"):
        video_mod.VideoModalityProcessor(custom_preprocessor_path="dummy")


def test_video_processor_raises_without_torchvision(monkeypatch):
    monkeypatch.setattr(video_mod, "_TORCHVISION_AVAILABLE", False)
    with pytest.raises(ImportError, match="torchvision"):
        video_mod.VideoModalityProcessor()


def test_image_processor_raises_without_cv2(monkeypatch):
    monkeypatch.setattr(img_mod, "_CV2_AVAILABLE", False)
    with pytest.raises(ImportError, match="opencv-python"):
        img_mod.ImageModalityProcessor(normalize_image=False)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def test_pose2text_dataset_raises_without_pose_format(monkeypatch):
    monkeypatch.setattr(pose2text_mod, "_POSE_FORMAT_AVAILABLE", False)
    with pytest.raises(ImportError, match="pose-format"):
        pose2text_mod.Pose2TextDataset()


def test_signwriting_dataset_raises_without_signwriting(monkeypatch):
    monkeypatch.setattr(sw_dataset_mod, "_SIGNWRITING_AVAILABLE", False)
    with pytest.raises(ImportError, match="signwriting"):
        sw_dataset_mod.SignWritingDataset()


def test_video2text_dataset_raises_without_av(monkeypatch):
    monkeypatch.setattr(video2text_mod, "_AV_AVAILABLE", False)
    with pytest.raises(ImportError, match="av"):
        video2text_mod.Video2TextDataset()


def test_video2text_dataset_raises_without_torchvision(monkeypatch):
    monkeypatch.setattr(video2text_mod, "_TORCHVISION_AVAILABLE", False)
    with pytest.raises(ImportError, match="torchvision"):
        video2text_mod.Video2TextDataset()


def test_video_processor_no_raise_without_cv2_when_no_custom_preprocessor(monkeypatch):
    monkeypatch.setattr(video_mod, "_CV2_AVAILABLE", False)
    # cv2 is only required when custom_preprocessor_path is set; without it the
    # processor should instantiate without error even if cv2 is absent.
    video_mod.VideoModalityProcessor(custom_preprocessor_path=None)

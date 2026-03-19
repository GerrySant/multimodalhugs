"""
Top-level pytest configuration.

Generates path-dependent TSV metadata files (video, pose, features) with
correct absolute paths for the current machine before any test runs.
These TSVs are not committed to git because they contain absolute paths.
"""

import os
import pytest

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@pytest.fixture(scope="session", autouse=True)
def generate_asset_tsvs():
    """Generate path-dependent TSV metadata files once per session."""
    from assets.generate_assets import (
        generate_video_tsv,
        generate_pose_tsv,
        generate_features_tsv,
        save_clip_image_processor,
    )
    save_clip_image_processor()
    generate_video_tsv()
    generate_pose_tsv()
    generate_features_tsv()

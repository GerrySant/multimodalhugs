"""
Asset generator for multimodalhugs test suite.

Run this script once on any machine after cloning the repository to:
  1. Download and save the CLIP image processor config (no model weights —
     just the JSON preprocessing config needed to resize video frames).
     This is committed to git so subsequent runs never need a network call.
  2. Generate the TSV metadata files for path-dependent modalities
     (video, pose, features).  These TSVs are NOT committed because they
     contain absolute paths.

Text and image metadata TSVs are committed directly (inline signal strings).

Usage:
    python tests/assets/generate_assets.py
"""

import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP_PROCESSOR_DIR = os.path.join(ASSETS_DIR, "processors", "clip_image_processor")


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write("\t".join(str(c) for c in row) + "\n")
    print(f"  Written: {path}")


def generate_video_tsv():
    video_dir = os.path.join(ASSETS_DIR, "video")
    write_tsv(
        os.path.join(video_dir, "metadata.tsv"),
        "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput",
        [
            (os.path.join(video_dir, "sample_01.mp4"), 0, 0, "__asl__", "__en__", "Let's open Access."),
            (os.path.join(video_dir, "sample_02.mp4"), 0, 0, "__asl__", "__en__", "Good."),
        ],
    )


def generate_pose_tsv():
    pose_dir = os.path.join(ASSETS_DIR, "pose")
    write_tsv(
        os.path.join(pose_dir, "metadata.tsv"),
        "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput",
        [
            (os.path.join(pose_dir, "sample_01.pose"), 0, 0, "__asl__", "__en__", "Let's open Access."),
            (os.path.join(pose_dir, "sample_02.pose"), 0, 0, "__asl__", "__en__", "Good."),
        ],
    )


def generate_features_tsv():
    features_dir = os.path.join(ASSETS_DIR, "features")
    write_tsv(
        os.path.join(features_dir, "metadata.tsv"),
        "signal\tsignal_start\tsignal_end\tencoder_prompt\tdecoder_prompt\toutput",
        [
            (os.path.join(features_dir, "sample_01.npy"), 0, 0, "__asl__", "__en__", "Let's open Access."),
            (os.path.join(features_dir, "sample_02.npy"), 0, 0, "__asl__", "__en__", "Good."),
        ],
    )


def save_clip_image_processor():
    """
    Download CLIPImageProcessor from HuggingFace and save only the preprocessing
    config (no tokenizer, no model weights — just the JSON files needed to resize
    and normalise frames before batching).  The saved files are committed to git
    so this download never happens again on subsequent machines.
    """
    sentinel = os.path.join(CLIP_PROCESSOR_DIR, "preprocessor_config.json")
    if os.path.exists(sentinel):
        print(f"  Already exists, skipping: {CLIP_PROCESSOR_DIR}")
        return

    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    os.makedirs(CLIP_PROCESSOR_DIR, exist_ok=True)
    processor.save_pretrained(CLIP_PROCESSOR_DIR)
    print(f"  Saved: {CLIP_PROCESSOR_DIR}")


if __name__ == "__main__":
    print("Saving CLIP image processor config (downloads once, then cached)...")
    save_clip_image_processor()

    print("Generating TSV metadata files for path-dependent modalities...")
    generate_video_tsv()
    generate_pose_tsv()
    generate_features_tsv()
    print("Done. Text and image TSVs are committed to git and do not need regeneration.")

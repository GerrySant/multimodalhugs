#!/usr/bin/env python3
"""
Script to extract sentence-level video clips from raw videos based on timestamps
provided in a CSV file. Clips are saved to the output directory, and a dataset TSV
with video paths and sentences is generated.

Usage:
    python create_realigned_clips.py \
        --csv_path path/to/realigned.csv \
        --video_dir path/to/raw_videos \
        --output_dir path/to/output_clips \
        --dataset_tsv path/to/output_dataset.tsv
"""

import os
import argparse
import pandas as pd
import subprocess
import multiprocessing
from tqdm import tqdm

def get_duration(video_path: str) -> float:
    """
    Get the duration of a video in seconds using ffprobe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Duration of the video in seconds. Returns 0.0 if error.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

def make_clip(row: dict, output_dir: str):
    """
    Create a clip from a video using ffmpeg, given start and end timestamps.

    Args:
        row (dict): Row with keys: 'path', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED'.
        output_dir (str): Directory where the clip will be saved.
    """
    src = row['path']
    dst = os.path.join(output_dir, f"{row['SENTENCE_NAME']}.mp4")
    start, end = row["START_REALIGNED"], row["END_REALIGNED"]
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", src,
        "-c", "copy",
        dst
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_dataset(csv_path: str, video_dir: str, output_dir: str, dataset_tsv: str):
    """
    Main processing pipeline: reads CSV, validates videos, creates clips, and writes TSV.

    Args:
        csv_path (str): Path to the CSV file with sentence-level annotations.
        video_dir (str): Directory containing raw video files.
        output_dir (str): Directory where clips will be saved.
        dataset_tsv (str): Path to output TSV with video clip paths and sentences.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path, sep="\t")

    # Build full video path column and keep only existing files
    df["path"] = df["VIDEO_NAME"].apply(lambda v: os.path.join(video_dir, f"{v}.mp4"))
    df = df[df["path"].apply(os.path.isfile)].copy()

    # Compute durations
    unique_paths = df["path"].unique().tolist()
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        durations_list = list(
            tqdm(pool.imap(get_duration, unique_paths),
                 total=len(unique_paths),
                 desc="Getting video durations")
        )
    durations = dict(zip(unique_paths, durations_list))
    df["VIDEO_DURATION"] = df["path"].map(durations)

    # Filter rows with valid end time
    df_valid = df[df["END_REALIGNED"] <= df["VIDEO_DURATION"]].copy()

    # Create dataset TSV
    df_valid["clip_path"] = df_valid["SENTENCE_NAME"].apply(
        lambda s: os.path.join(output_dir, f"{s}.mp4")
    )
    df_valid[["clip_path", "SENTENCE"]].to_csv(
        dataset_tsv, sep="\t", index=False, header=["video_path", "sentence"]
    )

    # Create clips
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        list(
            tqdm(
                pool.imap(lambda row: make_clip(row, output_dir), df_valid.to_dict('records')),
                total=len(df_valid),
                desc="Creating clips"
            )
        )

    print(f"\n✅ Dataset TSV written to: {dataset_tsv}")

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract sentence-level clips from videos and generate dataset TSV."
    )
    parser.add_argument("--csv_path", required=True, help="Path to realigned CSV file provided in How2sign dataset.")
    parser.add_argument("--video_dir", required=True, help="Directory with How2Sign raw videos.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the new aligned clips.")
    parser.add_argument("--dataset_tsv", required=True, help="Path to save output TSV containg 'video_path'	and 'sentence' for each sample.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        dataset_tsv=args.dataset_tsv
    )

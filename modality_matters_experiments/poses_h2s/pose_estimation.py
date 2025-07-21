#!/usr/bin/env python3
"""
Video-to-Pose Batch Processor
-----------------------------

This script processes a directory of `.mp4` videos and extracts poses for each video
using the `video_to_pose` tool with MediaPipe configuration. It traverses the input directory
recursively and saves the corresponding `.pose` files in the output directory, preserving
the relative folder structure.

Processed videos that already have a `.pose` output are skipped.

Usage:
    python3 script.py --input-dir <input_directory> --output-dir <output_directory>

Arguments:
    --input-dir   Path to the root directory containing the input videos (.mp4)
    --output-dir  Path to the root directory where output `.pose` files will be saved

Example:
    python3 script.py \
        --input-dir /data/videos \
        --output-dir /data/poses

The number of CPUs used is automatically detected (or taken from SLURM_CPUS_PER_TASK if running in SLURM).
"""

import os
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

def process_video(video_path, input_dir, output_dir):
    """
    Run video_to_pose on a single video and save the output pose file.

    Args:
        video_path (str): Path to the input video.
        input_dir (str): Root input directory (for relative path computation).
        output_dir (str): Root output directory.
    """
    rel = os.path.relpath(video_path, input_dir)
    base, _ = os.path.splitext(rel)
    out_path = os.path.join(output_dir, base + '.pose')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        'video_to_pose', '--format', 'mediapipe',
        '-i', video_path,
        '-o', out_path,
        '--additional-config', 'model_complexity=2,refine_face_landmarks=True'
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return video_path

def gather_videos(input_dir, output_dir):
    """
    Scan the input directory for `.mp4` files and determine which need processing.

    Args:
        input_dir (str): Root input directory.
        output_dir (str): Root output directory.

    Returns:
        tuple: (all_videos, to_process) where both are lists of video file paths.
    """
    videos = [
        os.path.join(root, fn)
        for root, _, fns in os.walk(input_dir)
        for fn in fns if fn.lower().endswith('.mp4')
    ]

    done = set()
    for root, _, fns in os.walk(output_dir):
        for fn in fns:
            if fn.lower().endswith('.pose'):
                done.add(os.path.splitext(os.path.relpath(fn, output_dir))[0])

    to_process = []
    for v in videos:
        rel = os.path.relpath(v, input_dir)
        base, _ = os.path.splitext(rel)
        if base not in done:
            to_process.append(v)

    return videos, to_process

def main():
    parser = argparse.ArgumentParser(description="Batch pose extraction from videos using video_to_pose.")
    parser.add_argument("--input-dir", required=True, help="Path to the input videos directory")
    parser.add_argument("--output-dir", required=True, help="Path to the output poses directory")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    print(f"Using {n_cpus} CPUs")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    videos, to_process = gather_videos(input_dir, output_dir)

    print(f"Total videos: {len(videos)}, remaining: {len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    worker = partial(process_video, input_dir=input_dir, output_dir=output_dir)
    with Pool(n_cpus) as pool:
        for _ in tqdm(pool.imap_unordered(worker, to_process),
                      total=len(to_process),
                      desc='Pose extraction'):
            pass

if __name__ == '__main__':
    main()

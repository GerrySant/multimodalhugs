#!/usr/bin/env python3
"""
Extract I3D features from one or more videos using a pretrained I3D model.

This script takes a video file (.mp4, .avi, .mov, .mkv) or a directory of such videos,
applies resizing, cropping, normalization, and splits into clips with a sliding window.
It then passes the clips through an I3D model and saves the extracted features
to .npy files.

Features are saved per-video under the specified output path.

Usage example:
    python extract_i3d_features.py \
        --checkpoint_path /path/to/checkpoint.pth.tar \
        --video_path /path/to/video_or_dir \
        --output_path /path/to/output_dir \
        --device cuda
"""

import os
import math
import sys
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ensure shared strategy for multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# append repo root to sys.path to find local modules
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

import models
from utils.misc import to_torch
from utils.imutils import im_to_numpy, im_to_torch, resize_generic
from utils.transforms import color_normalize

import shutil

def load_rgb_video(video_path: Path, fps: int) -> torch.Tensor:
    """
    Load a video file and return its frames as a torch.Tensor in RGB format.

    If the video frame rate does not match `fps`, a temporary copy is created using
    ffmpeg to enforce the desired fps.

    Args:
        video_path (Path): Path to the video file.
        fps (int): Target frame rate (frames per second).

    Returns:
        torch.Tensor: Video frames in shape (3, n_frames, height, width), normalized to [0,1].
    """
    cap = cv2.VideoCapture(str(video_path))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    if cap_fps != fps:
        tmp_video_path = f"{video_path}.tmp.{video_path.suffix}"
        shutil.move(video_path, tmp_video_path)
        cmd = (f"ffmpeg -i {tmp_video_path} -pix_fmt yuv420p "
               f"-filter:v fps=fps={fps} {video_path}")
        print(f"Generating new copy of video with frame rate {fps}")
        os.system(cmd)
        Path(tmp_video_path).unlink()
        cap = cv2.VideoCapture(str(video_path))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap_fps == fps, f"ffmpeg failed to produce a video at {fps}"

    rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (cv2) to RGB
        frame = frame[:, :, [2, 1, 0]]
        rgb_t = im_to_torch(frame)
        rgb.append(rgb_t)
    cap.release()

    rgb = torch.stack(rgb).permute(1, 0, 2, 3)
    print(f"Loaded video {video_path} with {rgb.shape[1]} frames [{cap_height}x{cap_width}] at {cap_fps}fps")
    return rgb


def prepare_input(
    rgb: torch.Tensor,
    resize_res: int = 224,
    inp_res: int = 224,
    mean: torch.Tensor = 0.5 * torch.ones(3),
    std: torch.Tensor = 1.0 * torch.ones(3),
) -> torch.Tensor:
    """
    Preprocess video frames for I3D input: resize, crop, normalize.

    Args:
        rgb (torch.Tensor): Raw RGB video tensor (3, T, H, W).
        resize_res (int): Resize target resolution.
        inp_res (int): Final input crop size.
        mean (torch.Tensor): Mean per channel for normalization.
        std (torch.Tensor): Std per channel for normalization.

    Returns:
        torch.Tensor: Preprocessed video tensor.
    """
    iC, iF, _, _ = rgb.shape
    rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
    for t in range(iF):
        tmp = rgb[:, t, :, :]
        tmp = resize_generic(
            im_to_numpy(tmp), resize_res, resize_res, interp="bilinear", is_flow=False
        )
        rgb_resized[t] = tmp
    rgb = np.transpose(rgb_resized, (3, 0, 1, 2))

    # center crop
    ulx = int((resize_res - inp_res) / 2)
    uly = int((resize_res - inp_res) / 2)
    rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
    rgb = to_torch(rgb).float()
    assert rgb.max() <= 1
    rgb = color_normalize(rgb, mean, std)
    return rgb


def sliding_windows(
    rgb: torch.Tensor,
    num_in_frames: int,
    stride: int,
) -> tuple:
    """
    Split video into sliding window clips.

    Args:
        rgb (torch.Tensor): Video tensor (3, T, H, W).
        num_in_frames (int): Number of frames per clip.
        stride (int): Step between consecutive clips.

    Returns:
        tuple: (clips tensor of shape (N, 3, num_in_frames, H, W), mid-frame indices as np.array)
    """
    C, nFrames, H, W = rgb.shape

    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
    print(f"{num_clips} clip{'s' if num_clips > 1 else ''} resulted from sliding window processing.")

    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
    t_mid = []
    for j in range(num_clips):
        actual_clip_length = min(num_in_frames, nFrames - j * stride)
        if actual_clip_length == num_in_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - num_in_frames
        t_mid.append(t_beg + num_in_frames / 2)
        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
    return rgb_slided, np.array(t_mid)


def load_model(
    checkpoint_path: Path,
    num_in_frames: int,
    endpoint: str,
    device: str,
    num_classes: int
) -> torch.nn.Module:
    """
    Load an I3D model and its checkpoint.

    Args:
        checkpoint_path (Path): Path to model checkpoint.
        num_in_frames (int): Number of frames per clip.
        endpoint (str): Model endpoint layer to extract.
        device (str): Device to load model on ("cpu" or "cuda").
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Loaded and ready I3D model.
    """
    print(f"Initializing I3D model with endpoint '{endpoint}' on CPU...", flush=True)
    model = models.InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint=endpoint,
        name='inception_i3d',
        in_channels=3,
        dropout_keep_prob=1.0,
        num_in_frames=num_in_frames,
        include_embds=True,
    )
    model = torch.nn.DataParallel(model)

    print(f"Loading checkpoint from {checkpoint_path} to CPU...", flush=True)
    ckpt = torch.load(str(checkpoint_path), map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract I3D features from video(s).')
    parser.add_argument('--checkpoint_path', type=Path, required=True,
                        help='Path to I3D checkpoint (.pth.tar)')
    parser.add_argument('--endpoint', type=str, default='AvgPool',
                        help='I3D endpoint to extract')
    parser.add_argument('--video_path', type=Path, required=True,
                        help='Path to input video file or directory')
    parser.add_argument('--output_path', type=Path, required=True,
                        help='Path to output .npy file or directory')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Target frame rate')
    parser.add_argument('--stride', type=int, default=8, help='Sliding window stride')
    parser.add_argument('--num_in_frames', type=int, default=64,
                        help='Frames per clip')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device: "cuda" or "cpu"')
    parser.add_argument('--num_classes', type=int, default=400,
                        help='Number of model classes')
    args = parser.parse_args()

    input_is_dir = args.video_path.is_dir()
    output_is_dir = args.output_path.suffix == '' or args.output_path.is_dir()

    if input_is_dir and not output_is_dir:
        raise ValueError("When --video_path is a directory, --output_path must also be a directory.")
    if output_is_dir and not args.output_path.exists():
        args.output_path.mkdir(parents=True)

    # collect video files
    if input_is_dir:
        vid_patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for pat in vid_patterns:
            video_files.extend(sorted(args.video_path.glob(pat)))
    else:
        video_files = [args.video_path]

    # skip already processed videos
    if input_is_dir:
        processed = set(p.stem for p in args.output_path.glob('*.npy'))
        original_count = len(video_files)
        video_files = [vf for vf in video_files if vf.stem not in processed]
        skipped = original_count - len(video_files)
        if skipped > 0:
            print(f"[INFO] Skipping {skipped} videos already processed.", flush=True)

    if not video_files:
        print("[INFO] No new videos to process. Exiting.", flush=True)
        sys.exit(0)

    model = load_model(args.checkpoint_path, args.num_in_frames,
                       args.endpoint, args.device, args.num_classes)

    error_videos = []

    for vid_path in tqdm(video_files, desc='Processing videos'):
        try:
            out_path = (args.output_path / vid_path.with_suffix('.npy').name) if output_is_dir else args.output_path

            rgb = load_rgb_video(vid_path, args.fps)
            rgb_inp = prepare_input(rgb)
            clips, _ = sliding_windows(rgb_inp, args.num_in_frames, args.stride)
            clips = clips.to(args.device)

            feats_list = []
            num_batches = math.ceil(len(clips) / args.batch_size)
            for i in range(num_batches):
                batch = clips[i*args.batch_size:(i+1)*args.batch_size]
                with torch.no_grad():
                    out = model(batch)
                feats = out if isinstance(out, torch.Tensor) else list(out.values())[-1].view(out[list(out.keys())[-1]].size(0), -1)
                feats_list.append(feats.cpu().numpy())
            features = np.concatenate(feats_list, axis=0)

            np.save(str(out_path), features)

        except Exception as e:
            print(f"[ERROR] Failed processing {vid_path}: {e}", flush=True)
            error_videos.append(str(vid_path))

    if error_videos:
        errors_file = (args.output_path / 'error_videos.txt') if output_is_dir else (args.output_path.parent / 'error_videos.txt')
        with open(errors_file, 'w') as f:
            for v in error_videos:
                f.write(v + '\n')
        print(f"\nList of videos with errors saved to {errors_file}")

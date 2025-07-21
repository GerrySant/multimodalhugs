# Poses Modality – Sign-to-Text Translation Experiments (SLTAT 2025)

This directory contains the scripts used to reproduce the pose-based sign-to-text translation experiments presented in the paper:  
**"Modality Matters: Training and Tokenization Effects in Sign-to-Text Translation"** (SLTAT 2025).

## Overview

The experiments in this folder use pose keypoints extracted from the [How2Sign dataset](https://how2sign.github.io/#download) as input. Pose keypoints are estimated using MediaPipe Holistic, reduced to focus on upper-body motion, and fed into the translation model through a linear projection into the M2M-100 encoder space. The models (Linear + M2M-100) are end2end trained.

All experiments were conducted using the [MultimodalHugs](https://github.com/GerardSoleCa/multimodalhugs) framework.

---

## 📄 Metadata File Preparation

Each training, validation, and test split must be described in a TSV (tab-separated values) metadata file. The metadata file should have the following columns:

| `signal` | `signal_start` | `signal_end` | `encoder_prompt` | `decoder_prompt` | `output` |
|----------|----------------|--------------|-------------------|-------------------|----------|
| `/path/to/sample.pose` | `0` | `0` | `__asl__` | `__en__` | `Sample target transcription.` |

- `signal`: Path to the `.pose` file (output of `pose_estimation.py`).
- `signal_start` / `signal_end`: Reserved for start/end timestamps (keep `0` if not needed or using video clips).
- `encoder_prompt`: Source language token (e.g., `__asl__`).
- `decoder_prompt`: Target language token (e.g., `__en__`).
- `output`: Ground truth transcription text.

Make sure to create separate metadata files for each split and update the paths accordingly in `config.yaml`.

---

## Files and Usage

### 🛠️ `config.yaml`

Main configuration file used for training and evaluation.  

Key notes:
- The fields `data.train_metadata_file`, `data.validation_metadata_file`, and `data.test_metadata_file` must be updated to point to your local `.tsv` metadata files.
- To switch between **original** and **aligned** versions of How2Sign, simply point these paths to metadata generated with either normal or realigned `.pose` files.
- To apply **uniform temporal downsampling**, set the value of `processor.skip_frames_stride` to 1 (no downsampling), 2, or 3 — as explored in the paper.  (You can play with any other integer value if desired)

---

### 🕺 `pose_estimation.py`

Script to extract pose keypoints from a directory of `.mp4` videos using MediaPipe Holistic and save them as `.pose` files.  
It recursively processes all videos and skips files already processed.

**Usage:**
```bash
python3 pose_estimation.py     --input-dir /path/to/videos     --output-dir /path/to/output_poses
```

Make sure [`pose-format`](https://github.com/sign-language-processing/pose) is installed and accessible, and adjust input/output paths to your setup.

---

### 🚀 `setup_and_train.sh`

Sets up the experiment and starts training.

**Usage:**
1. Update `ENV_PATH` to your Python environment.
2. Update `OUTPUT_PATH` to your desired checkpoint directory.

```bash
bash setup_and_train.sh
```

---

### 📈 `generate.sh`

Loads a trained model checkpoint and evaluates it using SacreBLEU and chrF.

**Usage:**
1. Set `ENV_PATH` to your environment.
2. Set `CKPT_PATH` to the checkpoint you want to evaluate.
3. Set `OUTPUT_PATH` to store the predictions.

```bash
bash generate.sh
```

---

## Notes

- All models use the same architecture: MediaPipe pose keypoints (offline) + linear adapter + M2M-100.
- Experiments were run on a single A100-80GB GPU using mixed-precision training.
- Batch sizes and gradient accumulation steps were adjusted per modality and per DS factor to maintain consistent effective batch sizes.
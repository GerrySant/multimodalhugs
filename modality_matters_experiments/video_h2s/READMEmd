# Video Modality – Sign-to-Text Translation Experiments (SLTAT 2025)

This directory contains the scripts used to reproduce the video-based sign-to-text translation experiments presented in the paper:  
**"Modality Matters: Training and Tokenization Effects in Sign-to-Text Translation"** (SLTAT 2025).

## Overview

The experiments in this folder use raw video frames from the [How2Sign dataset](https://how2sign.github.io/#download) as input. Videos are preprocessed and encoded using the CLIP visual encoder, then passed through a linear projection into the encoder space of the M2M-100 language model. The models (CLIP encoder + Linear + M2M-100) are end2end trained.

All experiments were conducted using the [MultimodalHugs](https://github.com/GerardSoleCa/multimodalhugs) framework.

---

## 📄 Metadata File Preparation

Each training, validation, and test split must be described in a TSV (tab-separated values) metadata file. The metadata file should have the following columns:

| `signal` | `signal_start` | `signal_end` | `encoder_prompt` | `decoder_prompt` | `output` |
|----------|----------------|--------------|-------------------|-------------------|----------|
| `/path/to/video_clip.mp4` | `0` | `0` | `__asl__` | `__en__` | `Sample target transcription.` |

- `signal`: Absolute path to the video clip file.
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
- To switch between **original** and **aligned** versions of How2Sign, simply point these paths to metadata generated with either normal or realigned video clips.
- To apply **uniform temporal downsampling**, set the value of `processor.skip_frames_stride` to 1 (no downsampling), 2, or 3 — as explored in the paper. (You can play with any other integer value if desired)

### 🎞️ `video_preprocessing.sh`

Script to crop and resize the original video clips to the format expected by the CLIP encoder (`224x224` resolution).  
Before training, run this script to generate preprocessed video clips.

**Important:**  
Update the paths inside this script to match your local setup for:
- The video input folders (aligned or not),
- The output location,
- The Python or conda environment.

You can run this script on a workstation or submit it to a SLURM cluster.

```bash
bash video_preprocessing.sh
```

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

- All video models use the same architecture: CLIP encoder + linear adapter + M2M-100.
- Experiments were run on a single A100-80GB GPU using mixed-precision training.
- Batch sizes and gradient accumulation steps were adjusted per modality and per DS factor to maintain consistent effective batch sizes.
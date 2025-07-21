# Features Modality – Sign-to-Text Translation Experiments (SLTAT 2025)

This directory contains the scripts used to reproduce the feature-based sign-to-text translation experiments presented in the paper:  
**"Modality Matters: Training and Tokenization Effects in Sign-to-Text Translation"** (SLTAT 2025).

## Overview

The experiments in this folder use precomputed I3D spatiotemporal features extracted from the [How2Sign dataset](https://how2sign.github.io/#download) as input. I3D features are a compact representation of video clips, capturing temporal dynamics while reducing computational cost. Features are projected into the encoder space of the M2M-100 language model through a linear adapter.

We experiment with two types of features:  
- Features we extracted ourselves using the [BSL1K I3D pretrained model](https://www.robots.ox.ac.uk/~vgg/research/bslattend/)  
- Features provided by the authors of [Duarte et al. (CVPR 2022)](https://ieeexplore.ieee.org/abstract/document/9879384), referred to as **features\*** in the paper. (These are not distributed here; you should request them from the authors.)

All experiments were conducted using the [MultimodalHugs](https://github.com/GerardSoleCa/multimodalhugs) framework.

---

## 📄 Metadata File Preparation

Each training, validation, and test split must be described in a TSV (tab-separated values) metadata file. The metadata file should have the following columns:

| `signal` | `signal_start` | `signal_end` | `encoder_prompt` | `decoder_prompt` | `output` |
|----------|----------------|--------------|-------------------|-------------------|----------|
| `/path/to/sample.npy` | `0` | `0` | `__asl__` | `__en__` | `Sample target transcription.` |

- `signal`: Path to the `.npy` file containing I3D features (output of feature extraction).
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
- To switch between **original** and **aligned** versions of How2Sign, simply point these paths to metadata generated with either normal or realigned `.npy` files.
- To apply **uniform temporal downsampling**, set the value of `processor.skip_frames_stride` to 1 (no downsampling), 2, or 3 — as explored in the paper.

---

### 🎬 Preprocessing: I3D Feature Extraction

To reproduce our extracted features (not features\*), you can use one of two methods:

✅ **Option 1: Use the official BSL1K repository**  
- Clone: https://github.com/gulvarol/bsl1k  
- Create a fresh Python environment and install dependencies as per their instructions.  
- Download the latest pretrained I3D checkpoint: [here](https://www.robots.ox.ac.uk/~vgg/research/bslattend/)  
- Use their scripts to extract I3D features.

✅ **Option 2: Use our simplified script**  
- Use `i3d_feature_extraction.py` in this folder.  
- This script performs the same pipeline in a more streamlined way.
- To process all splits automatically, use `extract_i3d_features.sh`. Adjust paths in the script to point to your environment, data, and output directories.

Run feature extraction on a single split manually:
```bash
python i3d_feature_extraction.py \
    --checkpoint_path /path/to/bsl5k.pth.tar \
    --video_path /path/to/videos \
    --output_path /path/to/output_npy_dir \
    --device cuda
```

Or run the full pipeline (train/val/test):

```bash
bash extract_i3d_features.sh
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

- Features* (adapted I3D features by Duarte et al.) are not included here; please contact the original authors to obtain them.
- All models use the same architecture: I3D features + linear adapter + M2M-100.
- Experiments were run on a single A100-80GB GPU using mixed-precision training.
- Batch sizes and gradient accumulation steps were adjusted per modality to maintain consistent effective batch sizes.
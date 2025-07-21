# Modality Matters Experiments – SLTAT 2025

This repository contains the code and instructions to replicate the experiments presented in the paper:  
**"Modality Matters: Training and Tokenization Effects in Sign-to-Text Translation"** (SLTAT 2025).

We study how input modality affects sign-to-text translation performance, comparing **video**, **pose keypoints**, and **I3D spatiotemporal features** as input representations.

All experiments use the [MultimodalHugs](https://github.com/GerardSoleCa/multimodalhugs) framework.

---

## 📚 Dataset

We use the [How2Sign dataset](https://how2sign.github.io/#download), in two versions:
- The standard clips distributed directly by the authors.
- The **realigned clips**, extracted from the original raw videos and the manually aligned annotations provided as CSV files inside the How2Sign dataset:  
  `How2Sign/sentence_level/<split>/text/en/raw_text/re_aligned/how2sign_realigned_<split>.csv`.

To create the realigned clips yourself, use the script:  
```bash
python modality_matters_experiments/create_realigned_clips.py \
    --csv_path /path/to/how2sign_realigned_train.csv \
    --video_dir /path/to/raw_videos \
    --output_dir /path/to/output_clips \
    --dataset_tsv /path/to/output_dataset.tsv
```

This script extracts sentence-level clips and generates a TSV file listing video paths and their corresponding transcriptions.

---

## 📂 Modalities

Each modality experiment has its own subdirectory with detailed instructions and scripts:

- 📹 [Video](/modality_matters_experiments/video_h2s/)  
- 🕺 [Poses](/modality_matters_experiments/poses_h2s/)  
- 🎞️ [Features](/modality_matters_experiments/features_h2s/)

Please refer to each subdirectory’s README for specifics on preprocessing, configuration, and training.

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{sant2025modality,
  title     = {Modality Matters: Training and Tokenization Effects in Sign-to-Text Translation},
  author    = {Gerard Sant and others},
  booktitle = {Proceedings of the 9th Workshop on Sign Language Translation and Avatar Technologies (SLTAT)},
  year      = {2025}
}
```

---

For questions or issues, feel free to open an issue on the [MultimodalHugs repository](https://github.com/GerardSoleCa/multimodalhugs).
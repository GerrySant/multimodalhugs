<div align="center">
  <h1>🎨 MultiModalHugs — *Modality Matters Branch*</h1>
</div>

This branch of **MultimodalHugs** contains the scripts, configurations, and instructions to reproduce the experiments from the paper:  

> **"Modality Matters: Training and Tokenization Effects in Sign-to-Text Translation"**  
> *Accepted to SLTAT 2025*

The experiments explore the impact of input modality and tokenization strategies on sign language to text translation, using the [How2Sign](https://how2sign.github.io/) dataset.

---

## 📄 Where to Start

We provide separate folders with scripts and configurations for each modality explored in the paper:

- `modality_matters_experiments/video_h2s`
- `modality_matters_experiments/poses_h2s`
- `modality_matters_experiments/features_h2s`

Additionally, the script `modality_matters_experiments/create_realigned_clips.py` allows you to generate the **realigned video clips** version of How2Sign used in our experiments.

For detailed instructions and guidance on replicating the experiments, please see:

👉 [**modality_matters_experiments/README.md**](modality_matters_experiments/README.md)

This README provides:
- An overview of the dataset versions used
- Instructions for preparing realigned clips
- Pointers to each modality’s specific README for modality-specific details

---

## ℹ️ About MultimodalHugs

**MultimodalHugs** is a lightweight, modular framework built on top of [Hugging Face](https://huggingface.co/) for training, evaluating, and deploying multimodal AI models.  

If you’re looking for the general-purpose framework documentation, please refer to the `main` branch of this repository.

---

## Citing this Work

If you use this code or the results from the *Modality Matters* paper, please cite:  

```bibtex
@misc{modalitymatters2025,
    title={Modality Matters: Training and Tokenization Effects in Sign-to-Text Translation},
    author={Sant, Gerard and Moryossef, Amit and Jiang, Zifan and Escolano, Carlos},
    year={2025},
    note={Submitted to SLTAT 2025}
}
```

And also cite MultimodalHugs as:

```bibtex
@misc{multimodalhugs2024,
    title={MultimodalHugs: A Reproducibility-Driven Framework for Multimodal Machine Translation},
    author={Sant, Gerard and Moryossef, Amit and Jiang, Zifan and Escolano, Carlos},
    howpublished={\url{https://github.com/GerrySant/multimodalhugs}},
    year={2024}
}

```

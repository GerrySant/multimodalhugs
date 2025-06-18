
# 🧩 How to Create a Custom Dataset in MultimodalHugs

MultimodalHugs provides a modular and extensible system for handling datasets across different modalities. This guide walks you through the process of defining and registering your own dataset class that integrates smoothly with the framework’s processors, configuration system, and training pipeline.

---

## 📁 1. Create a Dataset Configuration Class

Define a `@dataclass` that inherits from `MultimodalDataConfig` or `datasets.BuilderConfig` to configure modality-specific arguments, and preprocessing parameters.

```python
from multimodalhugs.data import MultimodalDataConfig, build_merged_omegaconf_config, gather_appropriate_data_cfg
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MyModalityDataConfig(MultimodalDataConfig):
    name: str = "MyModalityDataConfig"
    my_custom_param: Optional[str] = field(default=None, metadata={"help": "Your custom description here."})

    def __init__(self, cfg=None, **kwargs):
        data_cfg = gather_appropriate_data_cfg(cfg)
        valid_config, extra_args, cfg_for_super = build_merged_omegaconf_config(type(self), data_cfg, **kwargs)
        super().__init__(cfg=cfg_for_super, **extra_args)
        self.my_custom_param = valid_config.get("my_custom_param", self.my_custom_param)
```

---

## 🧠 2. Define the Dataset Class

Inherit from either:
- Any dataset class from HuggingFace `datasets` (e.g. `datasets.GeneratorBasedBuilder`) for a generic implementation,
- or from an existing dataset class like `BilingualText2TextDataset` if your use case extends a known modality.

Use the `@register_dataset("your_dataset_name")` decorator so the dataset becomes discoverable by the framework.

### 📌 Standardized Feature Fields in MultimodalHugs

MultimodalHugs encourages a **standard structure** for dataset features. Most datasets follow the schema:

```python
{
  "signal": str or np.ndarray,
  "signal_start": Optional[int],
  "signal_end": Optional[int],
  "encoder_prompt": Optional[str],
  "decoder_prompt": Optional[str],
  "output": Optional[str],
}
```

🛠️ You can **modify, add, or remove fields** as needed. However, doing so requires **careful consideration**:

- If you add a new field (e.g., `"signal_2"`, `"output_2"`, `"output_class"`), make sure that the processor you're using is updated accordingly to **read, process, or ignore** that field.
- If the processor does not expect the field, it might be ignored or cause errors during batching, tokenization, or training.
- If any of the above, you can easly create or adapt a processor following [this](customization/custom_processor_creation_guide.md) guidelines.

---

### 🧱 Dataset Class Template

```python
from multimodalhugs.utils.registry import register_dataset
from multimodalhugs.utils.utils import get_num_proc
from datasets import load_dataset, DatasetInfo, Features, SplitGenerator
from typing import Dict, Any
import datasets

@register_dataset("my_custom_dataset")
class MyCustomDataset(datasets.GeneratorBasedBuilder):
    def __init__(self, config: Optional[MyModalityDataConfig] = None, *args, **kwargs):
        config, kwargs = resolve_and_update_config(MyModalityDataConfig, config, kwargs)
        info = DatasetInfo(description="My custom dataset for XYZ modality.")
        super().__init__(info=info, *args, **kwargs)
        self.config = config

    def _info(self):
        features = {
            "signal": str,
            "signal_start": Optional[int],
            "signal_end": Optional[int],
            "encoder_prompt": Optional[str],
            "decoder_prompt": Optional[str],
            "output": Optional[str],
            # Example custom field
            "signal_2": Optional[str],
        }
        return DatasetInfo(
            description="My custom dataset for XYZ modality",
            features=datasets.Features(features),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        cfg = self.config
        return [
            SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"metafile_path": cfg.train_metadata_file, "split": "train"}),
            SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"metafile_path": cfg.validation_metadata_file, "split": "validation"}),
            SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"metafile_path": cfg.test_metadata_file, "split": "test"}),
        ]

    def _generate_examples(self, metafile_path: str, split: str):
        dataset = load_dataset("csv", data_files=[metafile_path], split="train", delimiter="\t", num_proc=get_num_proc())

        # Optional filtering or mapping
        # dataset = dataset.map(...).filter(...)

        for idx, item in enumerate(dataset):
            yield idx, {
                "signal": item["signal"],
                "signal_start": item.get("signal_start", 0),
                "signal_end": item.get("signal_end", 0),
                "encoder_prompt": item.get("encoder_prompt", ""),
                "decoder_prompt": item.get("decoder_prompt", ""),
                "output": item["output"],
                "signal_2": item.get("signal_2", None),
            }
```

---

## 🧪 3. Optional Preprocessing or Filtering

If your modality involves loading content from files (e.g., `.npy`, `.mp3`, `.mp4`), or needs preprocessing like duration filtering, apply `.map()` or `.filter()` operations:

```python
def process_sample(example):
    features = np.load(example["signal"])
    example["signal"] = features
    example["DURATION"] = features.shape[0]
    return example

dataset = dataset.map(process_sample, num_proc=get_num_proc())
```

And filtering utilities:
```python
from multimodalhugs.data import file_exists_filter, duration_filter

dataset = dataset.filter(lambda ex: file_exists_filter("signal", ex), num_proc=get_num_proc())

dataset = dataset.filter(lambda ex: duration_filter(self.config.max_frames, ex), num_proc=get_num_proc())
```

---

## ✅ 4. Register and Run Your Dataset

To activate your dataset:

1. Place your `.py` file inside the `multimodalhugs/data/` folder.
2. Make sure it’s imported in the `__init__.py` or another imported file so that `@register_dataset` is executed.

How to create an instance of your dataset:

```python
from multimodalhugs.training_setup.setup_utils import load_config, prepare_dataset
from multimodalhugs.data.datasets.bilingual_image2text import MyCustomDataset, MyModalityDataConfig

dataset_dir="/output_path/to/save/dataset/instance"

# Load cfg with dataset args from a yaml file:
cfg = load_config(config_path)

# Or create a pyhton dict containing datasets arguments:
cfg = {
    "train_metadata_file": </path/to/train.tsv>,
    "validation_metadata_file": </path/to/validation.tsv>,
    "test_metadata_file": </path/to/test.tsv>,
    "my_curstom_param": <value_for_my_custom_param>
    }

# Instantiate and prepare dataset, then save to disk
data_cfg = MyModalityDataConfig(cfg)
data_path = prepare_dataset(
    MyCustomDataset,
    data_cfg,
    dataset_dir
)
```

How to load the instance of your custom dataset:

```python
from datasets import load_from_disk
your_dataset = load_from_disk(dataset_dir)
```

---

## 📚 Examples to Check

Explore the existing datasets for inspiration:

| File | Modality | Description |
|------|----------|-------------|
| `pose2text.py` | Pose | Converts pose sequences into text |
| `video2text.py` | Video | Converts video segments to text |
| `bilingual_image2text.py` | Rendered Text | Uses images of text as input |
| `features2text.py` | Feature Sequences | Converts feature vectors into text |

---

## 🧾 Summary

| Step | Description |
|------|-------------|
| 1️⃣ | Define a custom `DataConfig` class using `MultimodalDataConfig` |
| 2️⃣ | Build a dataset class inheriting from a dataset class from HuggingFace `datasets` (e.g. `datasets.GeneratorBasedBuilder`) |
| 3️⃣ | Use `.map()` or `.filter()` for loading or validating content |
| 4️⃣ | Keep your `features` aligned with processor expectations |
| 5️⃣ | Register and run the dataset from the CLI |

---

Let us know if you'd like a template repo, processor class, or help debugging a custom dataset!


# 🔧 How to Create a Custom Model in MultimodalHugs

MultimodalHugs extends Hugging Face Transformers to support multimodal inputs (e.g., visual, pose, etc.) alongside text. This guide explains how to create, structure, register, and use a custom model in MultimodalHugs.

---

## 🗂️ 1. Folder Structure

To create a new model:

1. Inside `multimodalhugs/models/`, create a new folder with your model name (e.g., `my_custom_model`)
2. Add:
   - `modeling_my_custom_model.py` → Your model logic
   - `__init__.py` → Exposes model and config classes

3. Optionally, if your model uses a **custom configuration**, add:
   - `configuration_my_custom_model.py` → Custom config class

4. In all cases, you must define in `__init__.py`:
   - `MODEL_CLASS`: your model class
   - `CONFIG_CLASS`: either your custom class or a Hugging Face config class (e.g., `transformers.BartConfig`)
   - `CONFIG_NAME`: a string used to register the model

**Example structure:**

```
multimodalhugs/
└── models/
    ├── my_custom_model/
    │   ├── __init__.py
    │   ├── modeling_my_custom_model.py
    │   └── configuration_my_custom_model.py  # Optional
```

---

## 🧬 2. Implement Your Config Class (Optional)

If needed, define a subclass of `transformers.PretrainedConfig` in `configuration_my_custom_model.py`.

```python
from transformers import PretrainedConfig

class MyCustomModelConfig(PretrainedConfig):
    model_type = "my_custom_model"

    def __init__(self, my_param: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param
```

---

## 🧠 3. Implement Your Model Class

In `modeling_my_custom_model.py`, subclass `transformers.PreTrainedModel`.

Use `@register_model("my_custom_model")` to register it within MultimodalHugs.

```python
from transformers import PreTrainedModel
from multimodalhugs.utils.registry import register_model
from .configuration_my_custom_model import MyCustomModelConfig

@register_model("my_custom_model")
class MyCustomModel(PreTrainedModel):
    config_class = MyCustomModelConfig
    base_model_prefix = "my_custom_model"

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.my_param, config.my_param)

    def forward(self, input_features=None, **kwargs):
        return self.linear(input_features)
```

---

## 🔗 4. Link Components in `__init__.py`

Regardless of whether you define a new configuration or reuse an existing one (e.g., `BartConfig`), you must define the following variables:

```python
from .modeling_my_custom_model import MyCustomModel
from .configuration_my_custom_model import MyCustomModelConfig  # Or import from HF

MODEL_CLASS = MyCustomModel
CONFIG_CLASS = MyCustomModelConfig  # Or BartConfig, T5Config, etc.
CONFIG_NAME = "my_custom_model"
```

---

## 🔁 5. Auto-Registration Logic

MultimodalHugs automatically registers models by scanning the `models/` folder for subfolders that define:

- `MODEL_CLASS`
- `CONFIG_CLASS`
- `CONFIG_NAME`

This allows using your model with `AutoModel.from_pretrained(...)` as in Hugging Face.

---

## 🧩 6. Forward Method: Naming Conventions

MultimodalHugs encourages standardized naming for inputs:

| Modality         | Input Argument                         |
|------------------|----------------------------------------|
| Text             | `input_ids`, `attention_mask`, `input_embeds`          |
| General multimodal | `input_frames`, `input_embeds`, `<multimodal_input_related_masks>`                   |
| Prompt           | `encoder_prompt`, `encoder_prompt_length_padding_mask` (for models with encoder) |
| Output           | `decoder_input_ids`, `labels`          |

💡 Prefer using `input_frames` to represent any non-text modality.  
This promotes cleaner integration with data processors and general-purpose pipelines.

---

## 🧱 7. Optional: `build_model()` for Complex Logic

For complex initialization (e.g., loading pretrained backbones, extending vocab), define a `build_model()` method:

```python
@classmethod
def build_model(cls, **kwargs):
    config = cls.config_class.from_dict(kwargs)
    return cls(config)
```

MultimodalHugs will prefer this method over using standard `__init__()` initialization if it exists.

---

## ⚙️ 8. Model Setup & Usage

### 🚀 A. Setup a Custom Model Instance

If you want to **use your custom model with any of the pipelines** supported by MultimodalHugs, run:

```bash
multimodalhugs-setup --modality "<modality>" --config_path "<path_to_config>.yaml" --model
```

Otherwise, build and save the model programmatically:

```python
from omegaconf import OmegaConf
from multimodalhugs.training_setup.setup_utils import load_config, build_and_save_model_from_init

config_path = "</path/to/config_custom_model.yaml>"
output_dir = "</path/to/save/custom_model/instance>"
model_instance_name = "<name_of_the_model_instance>"

cfg = load_config(config_path)
model_cfg = OmegaConf.to_container(cfg.model if 'model' in cfg else cfg, resolve=True)

model_path = build_and_save_model_from_init(
    model_type=model_cfg.get("type"),
    config_path=config_path,
    output_dir=output_dir,
    run_name=model_instance_name
)
```

---

### 🤖 B. Use the Custom Model for Inference

```python
import torch
from transformers import AutoModel, AutoProcessor
from multimodalhugs.tasks.translation.inference_utils import batched_inference
import multimodalhugs.models  # <-- triggers auto-registration

# Paths
model_id = "/path/to/your/model/checkpoint"
processor_id = "/path/to/your/processor"
tsv_path = "/path/to/your/data.tsv"

# Setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model = AutoModel.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(processor_id)

# Run inference
output = batched_inference(model, processor, tsv_path, modality="<modality>", batch_size=8)
print(output["preds"])
```

---

## ✅ Final Checklist

✅ Your model folder lives in `multimodalhugs/models/`  
✅ It defines `MODEL_CLASS`, `CONFIG_CLASS`, `CONFIG_NAME`  
✅ (Recommended) You follow naming conventions like `input_frames`, `input_ids`, etc.  
✅ (Optional) You define `build_model()` for advanced setups  
✅ You use `multimodalhugs.models` to trigger auto-registration  
✅ When registered, you can always load your model with `transformers` `from_pretrained(...)`

---

## 📚 References

- [Hugging Face – Custom Models Guide](https://huggingface.co/docs/transformers/en/custom_models)
- [MultimodalHugs GitHub (internal)](https://github.com/GerrySant/multimodalhugs/tree/master)

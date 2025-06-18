# 🛠️ How to Create a Custom Processor in MultimodalHugs

MultimodalHugs allows for flexible multimodal processing through custom **processors**, which prepare multimodal inputs (e.g., video, image, pose, or features) into model-ready tensors. This guide walks you through the steps to create and register your own processor.

---

## 📚 1. What Is a Processor?

The processor acts as a transformation layer between the dataset and the model. It performs essential preprocessing steps tailored to each input modality, ensuring that raw multimodal signals are converted into structured representations compatible with the model. For example, it handles feature extraction, tokenization, and sequence alignment.

A processor in MultimodalHugs:

- Converts input data (e.g., image path, video tensor) into a tensor format. This is where the specific loading logic for each signal type is defined.
- Handles padding and masking.
- Optionally applies preprocessing logic (e.g., frame skipping, resizing).
- Inherits from `MultimodalSequence2SequenceProcessor` in the current version of MultimodalHugs. Future versions of the framework will use a more general base class to support broader use cases.

Each processor must implement:

- `_obtain_multimodal_input_and_masks()`
- Users can also define any number of custom `_obtain_*` methods to process new fields introduced by custom datasets, or to perform further processing on existing input signals. These methods are automatically detected and executed when calling the processor.
- Optionally `_transform_get_items_output()` for efficient `with_transform()` usage. The code inside this function is designed to be applied within the dataset logic via `.with_transform()`, allowing preprocessing such as decoding or conversion to tensors to be parallelized and efficiently executed before batching.

---

## 🧱 2. Base Class: `MultimodalSequence2SequenceProcessor`

Your custom processor should subclass:

```python
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor
```

You can override any of the base methods. Upon them, the recommended ones are:

- `__init__()`
- `_obtain_multimodal_input_and_masks()`
- `_transform_get_items_output()` *(recommended)*
- Any custom `_obtain_*` methods if needed. *(New ones can be also defined)*

This base class also handles prompts with `_obtain_encoder_prompt` and `_obtain_decoder_prompt`.

---

## ✨ 3. Implementing a New Processor

### Minimal Template

```python
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor
from multimodalhugs.data import pad_and_create_mask

class MyModalityProcessor(MultimodalSequence2SequenceProcessor):
    name = "my_modality_processor"
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _my_data_to_tensor(self, signal):
        # Implement modality-specific loading
        raise NotImplementedError

    def _obtain_multimodal_input_and_masks(self, batch, **kwargs):
        tensors = [self._my_data_to_tensor(sample["signal"]) for sample in batch]
        padded, masks = pad_and_create_mask(tensors)
        return {"input_frames": padded, "attention_mask": masks}, kwargs

    def _transform_get_items_output(self, batch):
        batch["signal"] = [self._my_data_to_tensor(x) for x in batch["signal"]]
        return batch
```

---

## 💡 4. Use `with_transform()` for Efficient Preprocessing

The `_transform_get_items_output()` method is applied early in the dataset pipeline using:

```python
dataset = dataset.with_transform(processor._transform_get_items_output)
```

This is useful to:

- Decode files from disk (video, .npy, etc)
- Apply early normalization, cropping
- Avoid IO inside `DataLoader` collate

Make sure to override this method in your processor!

---

## 🧰 5. Examples by Modality

Below are just a few examples of existing processors implemented in MultimodalHugs.

### 🔬 Features2Text

- Implements LRU cache for feature loading
- Applies frame skipping before padding

### 🖼️ Image2Text

- Supports string input as text-to-image rendering or image path
- Performs normalization using provided mean/std

### 🎞️ Video2Text

- Decodes from video files using OpenCV or `torchvision.read_video`
- Applies custom resizing or frame skipping
- Can flatten CHW if `join_chw=True`


> **Note:** These are not the only modalities supported—users are encouraged to create custom processors to suit their own data and task-specific needs.

---

## 🧪 6. Register and Use Your Processor

Once you have implemented your custom processor, follow these steps to register and use it:

### 🔨 Saving your custom processor instance

```python
from transformers import AutoProcessor, AutoTokenizer
from multimodalhugs.processors import MyModalityProcessor
from multimodalhugs.training_setup.setup_utils import save_processor

custom_processor_output_dir = "/directory/to/save/processor/instance"

# Load text tokenizer if needed
tokenizer_path = <path_to_tokenizer>
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Instantiate processor with modality-specific args
custom_processor_specific_args = {
    "specific_arg": <specific_arg_value>,
    # ... add more args as needed
}

custom_processor = MyModalityProcessor(
    tokenizer=tokenizer,
    **custom_processor_specific_args
)
save_processor(custom_processor, custom_processor_output_dir)
```

### 📥 Loading the custom processor instance

```python
from multimodalhugs.processors import MyModalityProcessor
from transformers import AutoProcessor, AutoTokenizer

# Register your processor class
MyModalityProcessor.register_for_auto_class()
AutoProcessor.register("my_modality_processor", MyModalityProcessor)

# Load the saved processor
processor = AutoProcessor.from_pretrained(
    processor_name_or_path=custom_processor_output_dir,
)
```

## ✅ Summary

| Step | Description                                         |   |
| ---- | --------------------------------------------------- | - |
| 1️⃣  | Subclass `MultimodalSequence2SequenceProcessor`     |   |
| 2️⃣  | Implement `_obtain_multimodal_input_and_mask()` or any extra `_obtain_*`needed.     |   |
| 3️⃣  | Optionally override `_transform_get_items_output()` |   |
| 4️⃣  | Handle new modalities using custom tensor loaders   |   |
| 5️⃣  | Register and test with CLI or training pipeline     |   |

---

Reach out if you'd like a pre-filled template or guidance on implementing a processor for your specific modality!


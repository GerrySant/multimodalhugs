# How to Create a Custom Processor

This guide covers three scenarios, from simplest to most involved:

- **A.** Compose existing modality processors into a new pipeline — no new classes needed.
- **B.** Implement a new `ModalityProcessor` for an unsupported modality.
- **C.** Create a named task processor (a `MultimodalMetaProcessor` subclass) that can be saved and loaded by name.

Before reading this guide, familiarise yourself with the [processor architecture overview](../processors/processors_overview.md).

---

## A. Compose existing processors

If your task uses modalities that are already supported, you can build a fully functional processor by composing `ProcessorSlot` objects into a `MultimodalMetaProcessor` directly — no subclassing required.

```python
from transformers import AutoTokenizer
from multimodalhugs.processors import (
    MultimodalMetaProcessor,
    ProcessorSlot,
    PoseModalityProcessor,
    TextModalityProcessor,
)

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

meta = MultimodalMetaProcessor(
    slots=[
        ProcessorSlot(
            processor=PoseModalityProcessor(reduce_holistic_poses=True),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
            column_map={
                "signal": "signal",
                "signal_start": "signal_start",
                "signal_end": "signal_end",
            },
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "decoder_prompt", "output": "output"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="encoder"),
            output_data_key="encoder_prompt",
            output_mask_key="encoder_prompt_length_padding_mask",
            column_map={"encoder_prompt": "signal"},
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="prompt"),
            output_data_key="decoder_input_ids",
            output_mask_key="decoder_attention_mask",
            column_map={"decoder_prompt": "signal"},
        ),
    ],
    tokenizer=tokenizer,
)
```

This instance can be saved and loaded immediately:

```python
meta.save_pretrained("/path/to/save/")
loaded = MultimodalMetaProcessor.from_pretrained("/path/to/save/")
```

### Multi-input example (video + pose → text)

Multiple encoder slots are supported — add them to the flat `slots` list. Each must use distinct TSV column names in its `column_map`:

```python
from multimodalhugs.processors import VideoModalityProcessor

meta = MultimodalMetaProcessor(
    slots=[
        ProcessorSlot(
            processor=VideoModalityProcessor(),
            output_data_key="video_frames",
            output_mask_key="video_attention_mask",
            column_map={
                "video_signal":       "signal",
                "video_signal_start": "signal_start",
                "video_signal_end":   "signal_end",
            },
        ),
        ProcessorSlot(
            processor=PoseModalityProcessor(),
            output_data_key="pose_frames",
            output_mask_key="pose_attention_mask",
            column_map={
                "pose_signal":       "signal",
                "pose_signal_start": "signal_start",
                "pose_signal_end":   "signal_end",
            },
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "decoder_prompt", "output": "output"},
        ),
    ],
    tokenizer=tokenizer,
)
```

### Non-text output example (image → image)

Any modality can be used as a label slot. Mark it with `is_label=True`:

```python
from multimodalhugs.processors import ImageModalityProcessor

meta = MultimodalMetaProcessor(
    slots=[
        ProcessorSlot(
            processor=ImageModalityProcessor(width=224, height=224),
            output_data_key="input_frames",
            output_mask_key="attention_mask",
        ),
        ProcessorSlot(
            processor=ImageModalityProcessor(width=224, height=224),
            output_data_key="label_frames",
            is_label=True,
            column_map={"label_image": "signal"},
        ),
    ],
    tokenizer=tokenizer,
)
```

---

## B. Implement a new `ModalityProcessor`

When your input modality is not covered by any built-in processor, subclass `ModalityProcessor` and implement the two-stage interface.

### Minimal template

```python
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from multimodalhugs.processors.modality_processor import ModalityProcessor
from multimodalhugs.data import pad_and_create_mask


class MyModalityProcessor(ModalityProcessor):
    """Processes <your modality> into tensors."""

    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param

    # ------------------------------------------------------------------
    # Stage 1: per-sample, called at dataset.with_transform() time
    # ------------------------------------------------------------------

    def process_sample(
        self,
        values: Union[Any, Dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor:
        """
        Load one sample and return a tensor.

        values is either:
          - a raw value (file path, string, ndarray, tensor, …) for single-column slots
          - a dict {processor_param_name: value} for multi-column slots

        If your modality needs no per-sample work (e.g. the data is already a tensor),
        the base class no-op is sufficient — do not override process_sample in that case.
        """
        if isinstance(values, torch.Tensor):
            return values          # already preprocessed; pass through
        # ... load / convert your data here ...
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Stage 2: per-batch, called inside the DataCollator
    # ------------------------------------------------------------------

    def process_batch(
        self,
        samples: List[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pad a list of [T_i, D] tensors to [B, T_max, D] and return a mask.
        Use pad_and_create_mask for sequences, or stack directly for fixed-size inputs.
        """
        padded, mask = pad_and_create_mask(samples)
        return padded, mask
```

### Tips

- `process_sample` runs in DataLoader worker processes — keep it thread-safe.
- `process_batch` has access to the full batch; do padding and stacking here.
- Use `pad_and_create_mask` (from `multimodalhugs.data`) for variable-length sequences. It returns `(padded_tensor [B, T_max, D], mask [B, T_max])`.
- For fixed-size inputs (e.g. a single image per sample), `torch.stack(samples)` is sufficient and no mask is needed (`return stacked, None`).
- Serialisation: `MultimodalMetaProcessor.save_pretrained()` serialises your processor's `__dict__` to JSON. Keep constructor parameters JSON-serialisable (strings, ints, floats, booleans, None). Private attributes (starting with `_`) and tokenizers are excluded automatically.

### Using your processor in a slot

```python
meta = MultimodalMetaProcessor(
    slots=[
        ProcessorSlot(
            processor=MyModalityProcessor(my_param=2.0),
            output_data_key="my_features",
            output_mask_key="my_attention_mask",
            # column_map defaults to {"signal": "signal"}
        ),
        ProcessorSlot(
            processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
            output_data_key="labels",
            is_label=True,
            column_map={"decoder_prompt": "decoder_prompt", "output": "output"},
        ),
    ],
    tokenizer=tokenizer,
)
```

For `from_pretrained()` to reconstruct your processor it must be importable from `multimodalhugs.processors`. See section C for how to register it.

---

## C. Create a named task processor

A named task processor is a `MultimodalMetaProcessor` subclass with a fixed slot configuration. This is the right choice when:

- You want a stable name for `AutoProcessor` registration.
- You expose task-specific constructor parameters (e.g. `reduce_holistic_poses`) and want them saved alongside the processor.
- You want users to call `MyTaskProcessor(tokenizer=tok, ...)` without constructing slots manually.

### Template

```python
import logging
from typing import Any, Optional

from multimodalhugs.processors.meta_processor import MultimodalMetaProcessor, ProcessorSlot
from multimodalhugs.processors.text_modality_processor import TextModalityProcessor
from mypackage.processors.my_modality_processor import MyModalityProcessor

logger = logging.getLogger(__name__)


class MyTask2TextProcessor(MultimodalMetaProcessor):
    # HF ProcessorMixin metadata
    name = "my_task2text_processor"
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        my_param: float = 1.0,
        **kwargs,
    ):
        # Pass-through for from_pretrained(), which reconstructs the processor by
        # calling cls(slots=..., tokenizer=...).
        # Detect this case by the presence of "slots" in kwargs and delegate
        # directly to the parent without rebuilding the slots.
        if "slots" in kwargs:
            super().__init__(tokenizer=tokenizer, **kwargs)
            return

        # Store task-specific params so they appear in __dict__ and are serialised.
        self.my_param = my_param

        super().__init__(
            slots=[
                ProcessorSlot(
                    processor=MyModalityProcessor(my_param=my_param),
                    output_data_key="my_features",
                    output_mask_key="attention_mask",
                    # column_map defaults to {"signal": "signal"}
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="label"),
                    output_data_key="labels",
                    is_label=True,
                    column_map={"decoder_prompt": "decoder_prompt", "output": "output"},
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="encoder"),
                    output_data_key="encoder_prompt",
                    output_mask_key="encoder_prompt_length_padding_mask",
                    column_map={"encoder_prompt": "signal"},
                ),
                ProcessorSlot(
                    processor=TextModalityProcessor(tokenizer=tokenizer, role="prompt"),
                    output_data_key="decoder_input_ids",
                    output_mask_key="decoder_attention_mask",
                    column_map={"decoder_prompt": "signal"},
                ),
            ],
            tokenizer=tokenizer,
        )
```

### The `slots` passthrough explained

`MultimodalMetaProcessor.from_pretrained()` reconstructs the processor by calling:

```python
cls(slots=[...], tokenizer=...)
```

Without the passthrough guard, the subclass `__init__` would try to build new slots from `my_param` and ignore the pre-built ones. The guard detects this reconstruction call and routes directly to `super().__init__`, preserving the deserialised slots.

### Registering with `AutoProcessor`

Registration must happen before any `save_pretrained()` or `from_pretrained()` call that uses your class name. The recommended place is your package's `__init__.py`:

```python
from transformers import AutoProcessor
from mypackage.processors.my_task2text_processor import MyTask2TextProcessor

MyTask2TextProcessor.register_for_auto_class("AutoProcessor")
AutoProcessor.register("my_task2text_processor", MyTask2TextProcessor)
```

After registration, `AutoProcessor.from_pretrained("/path/to/saved/")` will resolve to your class automatically.

### Save and load

```python
from transformers import AutoTokenizer
from mypackage.processors.my_task2text_processor import MyTask2TextProcessor

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
processor = MyTask2TextProcessor(tokenizer=tokenizer, my_param=2.0)
processor.save_pretrained("/path/to/save/")

# Load by class directly
loaded = MyTask2TextProcessor.from_pretrained("/path/to/save/")

# Load via AutoProcessor (requires registration above)
loaded = AutoProcessor.from_pretrained("/path/to/save/")
```

---

## Testing checklist

```python
import torch

# 1. process_sample returns a tensor
tensor = proc.process_sample("path/to/sample")
assert isinstance(tensor, torch.Tensor)

# 2. process_batch pads correctly and returns consistent shapes
t1 = torch.randn(10, 64)
t2 = torch.randn(15, 64)
data, mask = proc.process_batch([t1, t2])
assert data.shape == (2, 15, 64)
assert mask.shape == (2, 15)
assert mask[0, 10:].eq(0).all()    # padded positions are masked out

# 3. MetaProcessor __call__ returns expected keys
result = meta(batch_samples)
assert "my_features" in result
assert "attention_mask" in result
assert "labels" in result

# 4. Round-trip save / load is lossless
meta.save_pretrained(tmp_dir)
loaded = MultimodalMetaProcessor.from_pretrained(tmp_dir)
result_loaded = loaded(batch_samples)
assert torch.equal(result["labels"], result_loaded["labels"])
```

---

## Summary

| Scenario | What to do |
|---|---|
| New pipeline with existing modalities | Compose `ProcessorSlot` objects into a `MultimodalMetaProcessor(slots=[...])` |
| New modality | Subclass `ModalityProcessor`, implement `process_sample` and `process_batch` |
| Named task processor | Subclass `MultimodalMetaProcessor`, add the `slots` passthrough guard, register with `AutoProcessor` |

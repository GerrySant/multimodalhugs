# Fully Sharded Data Parallel (FSDP) with MultimodalHugs

This directory contains [Accelerate](https://huggingface.co/docs/accelerate) FSDP configuration files for running `multimodalhugs-train` and `multimodalhugs-generate` across multiple GPUs using PyTorch's Fully Sharded Data Parallel strategy.

## What is FSDP and when to use it

FSDP shards the model parameters, gradients, and optimizer states across all GPUs instead of replicating the full model on every device. This allows you to train models that are too large to fit on a single GPU.

Use FSDP when:
- Your model does not fit in the memory of a single GPU.
- You want to scale training across multiple GPUs on one or more machines.

For models that fit comfortably on a single GPU, standard `torchrun` DDP (the default) is simpler and usually faster.

## Prerequisites

```bash
pip install accelerate
```

Verify your accelerate installation:
```bash
accelerate env
```

## Provided configuration files

| File | Backbone | Wrap policy |
|---|---|---|
| `fsdp_m2m100.yaml` | M2M-100 | `TRANSFORMER_BASED_WRAP` — wraps `M2M100EncoderLayer`, `M2M100DecoderLayer`, `CLIPEncoderLayer` |
| `fsdp_t5.yaml` | T5 | `TRANSFORMER_BASED_WRAP` — wraps `T5Block`, `CLIPEncoderLayer` |
| `fsdp_size_based.yaml` | Any | `SIZE_BASED_WRAP` — wraps any submodule with ≥ 100M parameters |

**Choosing a policy:**
- `TRANSFORMER_BASED_WRAP` is the recommended policy. It wraps only the natural parallelism boundaries of the architecture (transformer blocks), which avoids splitting modules that must stay together. Use this whenever you know the backbone type.
- `SIZE_BASED_WRAP` is a generic fallback that works for any backbone. It is less precise and may produce suboptimal sharding, but requires no knowledge of the internal module names.

All configs default to:
- `FULL_SHARD` — maximum memory savings, all tensor kinds are sharded across GPUs.
- `bf16` mixed precision — change to `fp16` or `no` as your hardware requires.
- 4 GPUs on 1 machine — adjust `num_processes` to match your setup.

## Running training with FSDP

Replace `--config_file` with the config matching your backbone, and adjust `--num_processes` if needed.

```bash
# ----------------------------------------------------------
# 1. Specify paths
# ----------------------------------------------------------
export CONFIG_PATH="/path/to/your/config.yaml"
export OUTPUT_PATH="/path/to/your/output_directory"
export FSDP_CONFIG="examples/fsdp/fsdp_m2m100.yaml"   # or fsdp_t5.yaml / fsdp_size_based.yaml

# ----------------------------------------------------------
# 2. Launch training
# ----------------------------------------------------------
accelerate launch \
    --config_file $FSDP_CONFIG \
    --num_processes 4 \
    $(which multimodalhugs-train) \
    --task "translation" \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH
```

> **Note:** `$(which multimodalhugs-train)` resolves the entry-point script so that `accelerate launch` can spawn it correctly. If you are using a conda environment, make sure it is activated before running the command.

> **Tip:** You can override any field in the FSDP config directly on the command line, for example:
> ```bash
> accelerate launch --config_file $FSDP_CONFIG --mixed_precision fp16 --num_processes 8 ...
> ```

## Running generation/evaluation with FSDP

Generation with `multimodalhugs-generate` can also be distributed over multiple GPUs using the same configs.

```bash
export CONFIG_PATH="/path/to/your/config.yaml"
export CKPT_PATH="/path/to/checkpoint"
export PROCESSOR_PATH="/path/to/processor"
export DATA_PATH="/path/to/dataset"
export OUTPUT_PATH="/path/to/generate_output"
export FSDP_CONFIG="examples/fsdp/fsdp_m2m100.yaml"

accelerate launch \
    --config_file $FSDP_CONFIG \
    --num_processes 4 \
    $(which multimodalhugs-generate) \
    --task "translation" \
    --metric_name "sacrebleu" \
    --model_name_or_path $CKPT_PATH \
    --processor_name_or_path $PROCESSOR_PATH \
    --dataset_dir $DATA_PATH \
    --generate_output_dir $OUTPUT_PATH \
    --config_path $CONFIG_PATH
```

## Adapting the configs

If you are using a backbone other than M2M-100 or T5, update `fsdp_transformer_layer_cls_to_wrap` with the repeating transformer block class name of your backbone. You can find it by inspecting `model._no_split_modules` after loading the model:

```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("your/backbone")
print(model._no_split_modules)
```

For `MultiModalEmbedderModel`, the combined list of `_no_split_modules` from both the feature extractor (e.g., `CLIPEncoderLayer`) and the backbone is automatically assembled at initialization and available as `model._no_split_modules`.

## Directory overview

```
examples/fsdp/
├── README.md               # This file
├── fsdp_m2m100.yaml        # FSDP config for M2M-100 backbone
├── fsdp_t5.yaml            # FSDP config for T5 backbone
└── fsdp_size_based.yaml    # Generic size-based FSDP config (any backbone)
```


# 🔁 Resuming Training from a Checkpoint

In this guide, we explain how to continue a training experiment from a previous checkpoint using the [`multimodalhugs`](https://github.com/GerrySant/multimodalhugs/tree/master) framework.

---

## 🛠️ Initial Setup

We assume you already have a configuration file defined and accessible via the environment variable `CONFIG_PATH`.

We also assume you’ve already set up the training actors by running:

```bash
multimodalhugs-setup --modality "<modality>" --config_path $CONFIG_PATH
```

Then, you launched training with:

```bash
multimodalhugs-train --task "<task>" \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH
```

The `OUTPUT_PATH` directory will store all logs, checkpoints, predictions, and other outputs related to the training process.

---

## ⏯️ Resuming Training

Once training is complete (or interrupted), you may want to continue training for more steps. In that case, you have **two options**:

---

### ✅ Option 1: Resume from the last checkpoint automatically

```bash
multimodalhugs-train --task "translation" \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH \
    --overwrite_output_dir false \
    --max_steps 20000
```

If `overwrite_output_dir=false` (which is the default in Hugging Face), the trainer will **automatically detect the latest checkpoint** stored in `output_dir` and resume training from that point.

---

### 🎯 Option 2: Resume from a specific checkpoint manually

```bash
multimodalhugs-train --task "translation" \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_PATH \
    --resume_from_checkpoint $DESIRED_CHECKPOINT_PATH \
    --max_steps 20000
```

This option allows you to explicitly define the path to the checkpoint folder from which training should resume.

---

## ℹ️ Additional Notes
- This behavior is identical to how Hugging Face Transformers handles checkpoint resumption.
  
- You can safely modify any training hyperparameter when resuming (e.g., `--learning_rate`, `--max_steps`, etc.).

- You can also modify any argument related to the dataset, processor, or model (as long as the model changes do not alter the original architecture used during training).
  
- Training will continue from where it left off **based on the step stored in the checkpoint**. That means the resumed run will assume that `N` steps have already been performed. Therefore, make sure to adjust `--max_steps` accordingly to reflect the **total** number of steps you want (previous + new steps).
  
- If you're using a logger (like Weights & Biases), ensure that logging is properly resumed or restarted as needed.

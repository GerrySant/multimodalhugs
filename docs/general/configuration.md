# MultiModalHugs Configuration Documentation

The MultiModalHugs configuration file standardizes the setup for training, evaluating, and deploying multimodal models. It is divided into four main sections:

- **Model**
- **Data**
- **Processor**
- **Setup**
- **Training**

Each section is explained in detail below.

---

## Model Section

This section defines the model-related parameters.

- **model_type** (*Required*): Specifies the type of model to use (e.g., `"multimodal_embedder"`).
- **model_name_or_path** (*Optional*): If you already have a model instance created by `multimodalhugs-setup`, you can specify this field to load that model during training. If omitted, the `multimodalhugs-setup` command will automatically add it.

> **Note:** For additional model-specific settings, refer to the <a href="/docs/models/">models documentation</a>.
---

## Data Section

This section handles dataset configuration.

- **dataset_dir** (*General argument*): Points to the dataset instance created by `multimodalhugs-setup`. Although optional during setup, it is required when using `multimodalhugs-train` to load the dataset.

> **Note:** Specific dataset-related arguments depend on the dataset type. See the <a href="/docs/data/">data documentation</a> for details.

---

## Processor Section

This section defines processor-specific parameters.

- **processor_name_or_path** (*General argument*): Points to the processor instance created by `multimodalhugs-setup`. Like the dataset, this field is optional during setup but required for loading the processor during training.
> **Note:** Specific processor-related arguments depend on the processor type. 

---

## Setup Section

This section defines parameters for preparing the dataset, processor, and model. The `modality` and `config_path` arguments must be specified via the command line, while the following arguments can be included in the configuration file:

- **output_dir** (*Optional, default: None*): Specifies the base output directory for saving artifacts.
- **do_dataset** (*Optional, default: False*): If set to `true`, prepares the dataset during setup.
- **do_processor** (*Optional, default: False*): If set to `true`, sets up the processor during setup.
- **do_model** (*Optional, default: False*): If set to `true`, builds the model during setup.
- **update_config** (*Optional, default: None*): If set to `true`, writes the created artifact paths back into the configuration file.

> **Note:** The `modality` argument, which specifies the training setup modality (e.g., `"pose2text"`, `"signwriting2text"`, `"image2text"`, `"text2text"`, `"features2text"`, etc), and the `config_path` argument, which points to the YAML configuration file, are required and must be provided via the command line when running `multimodalhugs-setup`.

> **Note:** If all `do_*` arguments (`do_dataset`, `do_processor`, `do_model`) are set to `false`, the setup process will attempt to create all actors (dataset, processor, and model).

---

## Training Section

This section includes training-specific parameters. MultiModalHugs leverages Hugging Face's Trainer. You can refer to the official <a href="https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments">TrainingArguments documentation</a> for a complete list of configurable training parameters.

In addition to standard `TrainingArguments`, MultiModalHugs defines the following extra training parameters:

- **`worker_start_method`** (*Optional, default: None*): Multiprocessing start method for all worker processes. Accepted values: `"fork"`, `"spawn"`, `"forkserver"`. When `None`, the Python/OS default is used (`"fork"` on Linux, `"spawn"` on macOS/Windows).

  Set to `"spawn"` when using `backend="torchcodec"` in `VideoModalityProcessor` with `dataloader_num_workers > 0` on Linux. The `"fork"` start method cannot safely inherit a CUDA context into child processes and will cause errors or deadlocks. This setting is **global**: it affects DataLoader workers, `Dataset.map()` workers, and all other child processes spawned via `torch.multiprocessing`.

  ```yaml
  training:
    dataloader_num_workers: 4
    worker_start_method: spawn
  ```

- **`metric_name`** (*Optional, default: None*): Name of the metric to use for evaluation (any metric supported by `evaluate.load()`). Multiple metrics can be specified as a comma-separated string: `"bleu,wer"`.
- **`early_stopping_patience`** (*Optional, default: None*): Number of evaluation calls with no improvement after which training stops. Requires `load_best_model_at_end: true` and `metric_for_best_model` to be set.
- **`visualize_prediction_prob`** (*Optional, default: 0.05*): Fraction of evaluation samples for which predictions are logged.
- **`print_decoder_prompt_on_prediction`** (*Optional, default: False*): If `true`, includes the decoder prompt in logged predictions.
- **`print_special_tokens_on_prediction`** (*Optional, default: False*): If `true`, also logs a version of predictions that includes special tokens.

---

> **Summary:** Using this configuration file ensures reproducibility and reduces boilerplate code by standardizing the experimental setup across different multimodal tasks.


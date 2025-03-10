# MultiModalHugs Configuration Documentation

The MultiModalHugs configuration file standardizes the setup for training, evaluating, and deploying multimodal models. It is divided into four main sections:

- **Model**
- **Data**
- **Processor**
- **Training**

Each section is explained in detail below.

---

## Model Section

This section defines the model-related parameters.

- **model_type** (*Required*): Specifies the type of model to use (e.g., `"multimodal_embedder"`).
- **model_name_or_path** (*Optional*): If you already have a model instance created by `multimodalhugs-setup`, you can specify this field to load that model during training. If omitted, the `multimodalhugs-setup` command will automatically add it.

> **Note:** For additional model-specific settings, refer to the <a href="docs/models/">models documentation</a>.
> 
---

## Data Section

This section handles dataset configuration.

- **dataset_dir** (*General argument*): Points to the dataset instance created by `multimodalhugs-setup`. Although optional during setup, it is required when using `multimodalhugs-train` to load the dataset.

> **Note:** Specific dataset-related arguments depend on the dataset type. See the <a href="docs/data/">data documentation</a> for details.

---

## Processor Section

This section defines processor-specific parameters.

- **processor_name_or_path** (*General argument*): Points to the processor instance created by `multimodalhugs-setup`. Like the dataset, this field is optional during setup but required for loading the processor during training.

---

## Training Section

This section includes training-specific parameters. MultiModalHugs leverages Hugging Face's Trainer. You can refer to the official <a href="https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments">TrainingArguments documentation</a> for a complete list of configurable training parameters.

---

> **Summary:** Using this configuration file ensures reproducibility and reduces boilerplate code by standardizing the experimental setup across different multimodal tasks.


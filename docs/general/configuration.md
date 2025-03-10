# MultiModalHugs Configuration Documentation

<style>
  body {
    font-family: Arial, sans-serif;
  }
  .section-header {
    font-size: 1.5em;
    font-weight: bold;
    margin-top: 20px;
    padding-bottom: 5px;
    border-bottom: 2px solid #ddd;
  }
  .argument {
    font-weight: bold;
    color: #ff7b00;
  }
  .optional {
    color: #0066cc;
  }
  .required {
    color: #d32f2f;
  }
  .note {
    background-color: #f8f9fa;
    border-left: 5px solid #ff7b00;
    padding: 10px;
    margin: 15px 0;
  }
</style>

The MultiModalHugs configuration file standardizes the setup for training, evaluating, and deploying multimodal models. It is divided into four main sections:

- **Model**
- **Data**
- **Processor**
- **Training**

Each section is explained in detail below.

---

<div class="section-header">Model Section</div>

This section defines the model-related parameters.

- <span class="argument">model_type</span> (<span class="required">Required</span>): Specifies the type of model to use (e.g., `"multimodal_embedder"`).
- <span class="argument">model_name_or_path</span> (<span class="optional">Optional</span>): If you already have a model instance created by `multimodalhugs-setup`, you can specify this field to load that model during training. If omitted, the `multimodalhugs-setup` command will automatically add it.

<div class="note">
  <b>Note:</b> For additional model-specific settings, refer to the <a href="docs/models/">models documentation</a>.
</div>

---

<div class="section-header">Data Section</div>

This section handles dataset configuration.

- <span class="argument">dataset_dir</span> (<span class="required">General argument</span>): Points to the dataset instance created by `multimodalhugs-setup`. Although optional during setup, it is required when using `multimodalhugs-train` to load the dataset.

<div class="note">
  <b>Note:</b> Specific dataset-related arguments depend on the dataset type. See the <a href="docs/data/">data documentation</a> for details.
</div>

---

<div class="section-header">Processor Section</div>

This section defines processor-specific parameters.

- <span class="argument">processor_name_or_path</span> (<span class="required">General argument</span>): Points to the processor instance created by `multimodalhugs-setup`. Like the dataset, this field is optional during setup but required for loading the processor during training.

---

<div class="section-header">Training Section</div>

This section includes training-specific parameters. MultiModalHugs leverages Hugging Face's Trainer. You can refer to the official <a href="https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments">TrainingArguments documentation</a> for a complete list of configurable training parameters.

---

<div class="note">
  <b>Summary:</b> Using this configuration file ensures reproducibility and reduces boilerplate code by standardizing the experimental setup across different multimodal tasks.
</div>

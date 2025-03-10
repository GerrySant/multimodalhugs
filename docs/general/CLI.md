# MultiModalHugs Command-Line Interface (CLI) Documentation

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

## Introduction

MultiModalHugs is a framework designed to facilitate multimodal language modeling. It standardizes configuration into four main sections:

- <span class="argument">Model</span>: Defines model-specific parameters.
- <span class="argument">Data</span>: Defines dataset-specific settings.
- <span class="argument">Processor</span>: Defines preprocessing settings.
- <span class="argument">Training</span>: Defines training parameters.

Each section is explained in detail below.

---

<div class="section-header">MultiModalHugs Command-Line Interface (CLI) Documentation</div>

### multimodalhugs-setup

This command sets up the training environment by creating and saving the dataset and processor instances.

#### Usage:

```bash
multimodalhugs-setup --modality {pose2text,signwriting2text,image2text} --config_path CONFIG_PATH
```

#### Arguments:

- <span class="argument">--modality</span> (<span class="required">Required</span>): Specifies the modality (e.g., 'pose2text', 'signwriting2text', or 'image2text').
- <span class="argument">--config_path</span> (<span class="required">Required</span>): Path to the YAML configuration file.

<div class="note">
  <b>Note:</b> This command must be run before training to properly initialize dataset and processor instances.
</div>

#### Example Help Output:

```bash
usage: multimodalhugs-setup [-h] --modality {pose2text,signwriting2text,image2text} --config_path CONFIG_PATH

MultimodalHugs Setup CLI. Use --modality to select the training setup for a given modality.

options:
  -h, --help            show this help message and exit
  --modality {pose2text,signwriting2text,image2text}
                        Specify the modality (e.g. 'pose2text', 'signwriting2text', or 'image2text').
  --config_path CONFIG_PATH
                        Path to the configuration file.
```

---

### multimodalhugs-train

This command initiates training using Hugging Face's `Trainer`, supporting various training parameters.

#### Usage:

```bash
multimodalhugs-train --task {translation} [additional arguments...]
```

#### Arguments:

- <span class="argument">--task</span> (<span class="required">Required</span>): Specifies the training task (currently only "translation" is supported).
- Additional parameters such as learning rate, batch size, and number of epochs can be specified.

To view all available training options, run:

```bash
multimodalhugs-train --task translation --help
```

<div class="note">
  <b>Note:</b> Refer to the <a href="https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments">Hugging Face TrainingArguments documentation</a> for a complete list of training options.
</div>

#### Example Help Output:

```bash
usage: multimodalhugs-train [-h] --task {translation} [additional arguments...]

MultimodalHugs Training CLI. Use --task to define the training objective.

options:
  -h, --help            show this help message and exit
  --task {translation}  Specify the training task (currently only "translation" is supported).
  --config_path CONFIG_PATH
                        Path to the configuration file.
  --learning_rate LEARNING_RATE
                        Set the learning rate.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Set the number of training epochs.
```

---

### multimodalhugs-generate

This command is used for evaluation and generating predictions.

#### Usage:

```bash
multimodalhugs-generate --task {translation} [additional arguments...]
```

#### Arguments:

- <span class="argument">--task</span> (<span class="required">Required</span>): Specifies the evaluation task (currently only "translation" is supported).
- <span class="argument">--config_path</span> (<span class="required">Required</span>): Path to the YAML configuration file.
- <span class="argument">--model_name_or_path</span> (<span class="required">Required</span>): Path to the trained model.
- <span class="argument">--processor_name_or_path</span> (<span class="required">Required</span>): Path to the processor instance.
- <span class="argument">--dataset_dir</span> (<span class="required">Required</span>): Path to the dataset.
- <span class="argument">--output_dir</span> (<span class="required">Required</span>): Directory to save generated outputs.

To view all available options, run:

```bash
multimodalhugs-generate --task translation --help
```

#### Example Help Output:

```bash
usage: multimodalhugs-generate [-h] --task {translation} [additional arguments...]

MultimodalHugs Generation CLI. Use --task to specify the generation objective.

options:
  -h, --help            show this help message and exit
  --task {translation}  Specify the evaluation task (currently only "translation" is supported).
  --config_path CONFIG_PATH
                        Path to the configuration file.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to the trained model.
  --processor_name_or_path PROCESSOR_NAME_OR_PATH
                        Path to the processor instance.
  --dataset_dir DATASET_DIR
                        Path to the dataset.
  --output_dir OUTPUT_DIR
                        Directory to save generated outputs.
```

---

<div class="section-header">Conclusion</div>

MultiModalHugs CLI provides a streamlined workflow from setup to training and evaluation. Users can refer to the `--help` option in each command for further details and consult the official Hugging Face documentation for additional guidance.

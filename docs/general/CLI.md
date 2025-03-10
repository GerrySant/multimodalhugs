# MultiModalHugs Command-Line Interface (CLI) Documentation

## Introduction

MultiModalHugs is a framework designed to facilitate multimodal language modeling. It standardizes configuration into four main sections:

- **Model**: Defines model-specific parameters.
- **Data** Defines dataset-specific settings.
- **Processor**: Defines preprocessing settings.
- **Training**: Defines training parameters.

Each section is explained in detail below.

---

## MultiModalHugs Command-Line Interface (CLI) Documentation

### multimodalhugs-setup

This command sets up the training environment by creating and saving the dataset and processor instances.

#### Usage:

```bash
multimodalhugs-setup --modality {pose2text,signwriting2text,image2text} --config_path CONFIG_PATH
```

#### Arguments:

- **--modality** (*Required*): Specifies the modality (e.g., 'pose2text', 'signwriting2text', or 'image2text').
- **--config_path** (*Required*): Path to the YAML configuration file.

> **Note:** This command must be run before training to properly initialize dataset and processor instances.

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

- **--task** (*Required*): Specifies the training task (currently only "translation" is supported).
- Additional parameters such as learning rate, batch size, and number of epochs can be specified.

To view all available training options, run:

```bash
multimodalhugs-train --task translation --help
```

> **Note:** Refer to the [Hugging Face TrainingArguments documentation](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments) for a complete list of training options.


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

- **--task** (*Required*): Specifies the evaluation task (currently only "translation" is supported).
- **--config_path** (*Required*): Path to the YAML configuration file.
- **--model_name_or_path** (*Required*): Path to the trained model.
- **--processor_name_or_path** (*Required*): Path to the processor instance.
- **--dataset_dir** (*Required*): Path to the dataset.
- **--output_dir** (*Required*): Directory to save generated outputs.

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

## Conclusion

MultiModalHugs CLI provides a streamlined workflow from setup to training and evaluation. Users can refer to the `--help` option in each command for further details and consult the official Hugging Face documentation for additional guidance.

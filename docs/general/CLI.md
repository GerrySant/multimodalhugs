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

This command sets up the training environment by creating and saving the dataset, processor, and model instances.

#### Usage:

```bash
multimodalhugs-setup --modality {pose2text,signwriting2text,image2text,text2text,features2text,video2text} --config_path $CONFIG_PATH [additional arguments...]
```

#### Arguments:

- **--modality** (*Required*): Specifies the training setup modality. Options include: `pose2text`, `signwriting2text`, `image2text`, `text2text`, `features2text`, or `video2text`.
- **--config_path** (*Required*): Path to the YAML configuration file.
- **--output_dir** (*Optional, default: None*): Specifies the base output directory for saving artifacts.
- **--do_dataset** (*Optional, default: False*): If set to `true`, prepares the dataset during setup.
- **--do_processor** (*Optional, default: False*): If set to `true`, sets up the processor during setup.
- **--do_model** (*Optional, default: False*): If set to `true`, builds the model during setup.
- **--update_config** (*Optional, default: None*): If set to `true`, writes the created artifact paths back into the configuration file.

> **Note:** This command must be run before training to properly initialize dataset, processor, and model instances.

> **Note:** Apart from `--modality` and `--config_path`, which must be specified via the command line, all other arguments (`--do_dataset`, `--do_processor`, `--do_model`, `--output_dir`, `--update_config`) can also be defined in the YAML configuration file under the `setup` section. Command-line arguments will always take priority over those defined in the configuration file.

> **Note:** If all `do_*` arguments (`do_dataset`, `do_processor`, `do_model`) are set to `false`, the setup process will attempt to create all actors (dataset, processor, and model).

#### Example Help Output:

```bash
usage: multimodalhugs-setup [-h] --modality {pose2text,signwriting2text,image2text,text2text,features2text,video2text} --config_path CONFIG_PATH [--do_dataset [DO_DATASET]] [--do_processor [DO_PROCESSOR]] [--do_model [DO_MODEL]] [--output_dir OUTPUT_DIR] [--update_config UPDATE_CONFIG]

options:
  -h, --help            show this help message and exit
  --modality {pose2text,signwriting2text,image2text,text2text,features2text,video2text}
                        Training setup modality. (default: None)
  --config_path CONFIG_PATH
                        Path to YAML configuration file (default: None)
  --output_dir OUTPUT_DIR
                        Base output directory. (default: None)
  --do_dataset [DO_DATASET]
                        Prepare the dataset. (default: False)
  --do_processor [DO_PROCESSOR]
                        Set up the processor. (default: False)
  --do_model [DO_MODEL]
                        Build the model. (default: False)
  --update_config UPDATE_CONFIG
                        Write created artifact paths back into the config file. (default: None)
```

---

### multimodalhugs-train

This command initiates training using Hugging Face's `Trainer`, supporting various training parameters.

#### Usage:

```bash
multimodalhugs-train --task <task_name> --output_dir $OUTPUT_DIR [--config_path $CONFIG_PATH] [--setup_path $SETUP_PATH] [additional arguments...]
```

#### Arguments:

- **--task** (*Required*): Specifies the training task (currently only "translation" is supported).
- **--output_dir** (*Required, str, defaults to "trainer_output"*): The output directory where the model predictions and checkpoints will be written.
- **--config_path** (*Optional, str, default: None*): Path to the YAML configuration file. This file can contain arguments for all actors (model, dataset, processor) as well as training hyperparameters. Con only be specified via the command line.
- **--setup_path** (*Optional, str, default: None*): Path to the setup directory containing `actors_paths.yaml`. Used only if `model_name_or_path`, `processor_name_or_path`, or `dataset_dir` are not provided via the command line or configuration file. If not provided, the path is inferred from `training_args.output_dir/setup`.
- Additional parameters such as learning rate, batch size, and number of epochs can be specified. Refer to the [Hugging Face TrainingArguments documentation](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments) for a complete list of training options.

To view all available training options, run:

```bash
multimodalhugs-train --task <task_name> --help
```

> **Note:** Refer to the [Hugging Face TrainingArguments documentation](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments) for a complete list of training options.

#### Example Help Output:

```bash
usage: multimodalhugs-train [-h] --task {translation} --output_dir OUTPUT_DIR [--config_path CONFIG_PATH] [--setup_path SETUP_PATH] [additional arguments...]

MultimodalHugs Training CLI. Use --task to define the training objective.

options:
  -h, --help            show this help message and exit
  --task <task_name>    Specify the training task (currently only "translation" is supported).
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written (default: "trainer_output").
  --config_path CONFIG_PATH
                        Path to YAML config file (default: None).
  --setup_path SETUP_PATH
                        Path to the setup directory containing actors_paths.yaml, used only if model_name_or_path, processor_name_or_path, or dataset_dir are not provided via commandline or config file. In train, if not provided, the path is inferred from <training_args.output_dir>/setup (default: None).
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
multimodalhugs-generate --task <task_name> [additional arguments...]
```

#### Arguments:

- **--task** (*Required*): Specifies the evaluation task (currently only "translation" is supported).
- **--model_name_or_path** (*Required*):\* Path to the trained model. 
- **--processor_name_or_path** (*Required*):\* Path to the processor instance.
- **--dataset_dir** (*Required*):\* Path to the dataset.
- **--config_path** (*Optional*): Path to the YAML configuration file.
- **--output_dir** (*Required*):\* Directory to save generated outputs.
  
  > **\*** This field can be either specified in the config or as argument

  > **Note:** If the configuration file specified via `--config_path` contains any of the following arguments: `model_name_or_path`, `processor_name_or_path`, or `dataset_dir`, the respective command-line argument can be omitted.

To view all available options, run:

```bash
multimodalhugs-generate --task <task_name> --help
```

#### Example Help Output:

```bash
usage: multimodalhugs-generate [-h] --task <task_name> [additional arguments...]

MultimodalHugs Generation CLI. Use --task to specify the generation objective.

options:
  -h, --help            show this help message and exit
  --task {translation}  Specify the evaluation task (currently only "translation" is supported).
  --metric_name METRIC_NAME
                        Name of the metric to use (any metric supported by evaluate.load())
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to the trained model.
  --processor_name_or_path PROCESSOR_NAME_OR_PATH
                        Path to the processor instance.
  --dataset_dir DATASET_DIR
                        Path to the dataset.
  --config_path CONFIG_PATH
                        Path to the configuration file.
  --setup_path SETUP_PATH
                        Path to the setup directory containing actors_paths.yaml, used only if model_name_or_path, processor_name_or_path, or dataset_dir are not provided via commandline or config file. If not provided, pipeline tries to infer the path from <training_args.output_dir>/setup (default: None).
  --output_dir OUTPUT_DIR
                        Directory to save generated outputs.
```

---

## Conclusion

MultiModalHugs CLI provides a streamlined workflow from setup to training and evaluation. Users can refer to the `--help` option in each command for further details and consult the official Hugging Face documentation for additional guidance.

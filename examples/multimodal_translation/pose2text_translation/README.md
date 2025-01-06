# Pose2Text Translation Example

This directory shows how to prepare and train a model for the Pose2Text Translation task using the MultimodalHugs framework. Below, you will find step-by-step instructions for dataset preparation, setting up the training environment, and launching the training process.

## Dataset Preparation

To train a Pose2Text model, you need to create metadata files for each dataset partition (train, validation, and test). For the How2Sign dataset, an example preprocessing script is provided:

[Dataset Preprocessing Script](https://github.com/GerrySant/multimodalhugs/blob/master/examples/multimodal_translation/pose2text_translation/example_scripts/how2sign_dataset_preprocessing_script.py)

#### Metadata File Requirements

The `metadata.tsv` files for each partition must include the following fields:

- `input_pose`: Path to the pose input.
- `source_start`: Start timestamp of the input segment.
- `source_end`: End timestamp of the input segment.
- `input_clip`: Path to the input video clip.
- `input_text`: Original text transcription.
- `source_prompt`: Source prompt for input conditioning.
- `generation_prompt`: Prompt to guide text generation.
- `output_text`: Target text for translation.
  
If `input_clip` is provided, fields `input_pose`, `source_start` and `source_end` are not required. If `source_prompt` and `generation_prompt` are nout needed, can be let empty.


## Training Setup

The MultimodalHugs framework is built on top of HuggingFace, focusing on creating objects from autoclasses and using them in the training script.

For Pose2Text experiments, you can use the example script provided:

[Training Setup Script](https://github.com/GerrySant/multimodalhugs/blob/master/examples/multimodal_translation/pose2text_translation/example_scripts/how2sign_training_setup.py)

#### Key Components in the Script

1. **Configuration**:
   - Uses `OmegaConf` to load the training configuration from a `.yaml` file.

2. **Dataset Initialization**:
   - Loads the How2Sign dataset and prepares it for training.

3. **Tokenizer Setup**:
   - Loads the main tokenizer and adds any new special tokens from vocabulary files.

4. **Processor**:
   - Initializes the `Pose2TextTranslationProcessor` to preprocess pose data. This is responsible for constructing the batch elements from the list of samples.

5. **Model Building**:
   - Creates a `MultiModalEmbedderModel` instance using the loaded configuration and tokenizers.

6. **Saving Outputs**:
   - Saves the processor, model, and dataset paths as environment variables for use in training.

####  Running the Script

```bash
python how2sign_training_setup.py --config_path path/to/your_config.yaml
```

This script will output the following paths:

- `MODEL_PATH`: Path to the saved model.
- `PROCESSOR_PATH`: Path to the saved processor.
- `DATA_PATH`: Path to the prepared dataset.

## Training Command

Once the dataset and components are prepared, you can launch the training script. An example shell script is provided below:

```bash
#!/bin/bash

source /path/to/anaconda3/bin/activate multimodalhugs

export MODEL_NAME="pose2text_how2sign"
export REPO_PATH="/path/to/repositories"
export CONFIG_PATH="/path/to/your_config.yaml"
export OUTPUT_PATH="/path/to/output"
export CUDA_VISIBLE_DEVICES=0
export EVAL_STEPS=250

python $REPO_PATH/multimodalhugs/examples/multimodal_translation/run_translation.py \
    --model_name_or_path $MODEL_PATH \
    --processor_name_or_path $PROCESSOR_PATH \
    --run_name $MODEL_NAME \
    --dataset_dir $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --do_train True \
    --do_eval True \
    --logging_steps 100 \
    --remove_unused_columns False \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --eval_steps $EVAL_STEPS \
    --save_strategy "steps" \
    --save_steps $EVAL_STEPS \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model 'bleu' \
    --overwrite_output_dir \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --warmup_steps 20000 \
    --max_steps 200000 \
    --predict_with_generate True \
    --lr_scheduler_type "inverse_sqrt" \
    --report_to none
```

### Hyperparameters

The hyperparameters in this script align with those available in HuggingFace's `Seq2SeqTrainer`, offering flexibility for optimization and experimentation.

## Summary

By following the steps outlined in this README, you can:

1. Prepare the How2Sign dataset for Pose2Text Translation.
2. Configure and initialize the training components.
3. Launch and monitor the training process for your Pose2Text model.

For more details, consult the [MultimodalHugs Documentation](https://github.com/GerrySant/multimodalhugs).


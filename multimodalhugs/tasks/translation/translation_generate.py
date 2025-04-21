#!/usr/bin/env python
# coding=utf-8
"""
Script to evaluate a trained model on the test partition.

It maintains the necessary blocks to:
- Load arguments and configuration from the command line or a YAML file.
- Set up the evaluation environment (logging, telemetry, etc.).
- Load and preprocess the test dataset.
- Configure the model, tokenizer/processor, and data collator.
- Execute the evaluation and save the predictions and metrics.

The script allows the user to specify the metric to use (any metric supported by evaluate.load())
and retains the possibility to configure parameters via YAML, as in the training script.
"""

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    set_seed,
    GenerationConfig,
)
from multimodalhugs.processors import (
    SignwritingProcessor,
    Pose2TextTranslationProcessor,
    Image2TextTranslationProcessor,
    Text2TextTranslationProcessor,
    Features2TextTranslationProcessor
)
from multimodalhugs.models import MultiModalEmbedderModel, MultiModalEmbedderConfig
from multimodalhugs import MultiLingualSeq2SeqTrainer

# Register multimodal classes
AutoConfig.register("multimodal_embedder", MultiModalEmbedderConfig)
AutoModelForSeq2SeqLM.register(MultiModalEmbedderConfig, MultiModalEmbedderModel)

Pose2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("pose2text_translation_processor", Pose2TextTranslationProcessor)

Features2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("features2text_translation_processor", Features2TextTranslationProcessor)

SignwritingProcessor.register_for_auto_class()
AutoProcessor.register("signwritting_processor", SignwritingProcessor)

Image2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("image2text_translation_processor", Image2TextTranslationProcessor)

Text2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("text2text_translation_processor", Text2TextTranslationProcessor)


import logging
import os
import sys
import argparse

import datasets
import evaluate
import numpy as np
from datasets import load_from_disk

import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

from multimodalhugs.data import DataCollatorMultimodalSeq2Seq
from multimodalhugs.utils import print_module_details

from multimodalhugs.tasks.translation.config_classes import ModelArguments, ProcessorArguments, DataTrainingArguments, ExtraArguments, ExtendedSeq2SeqTrainingArguments
from multimodalhugs.tasks.translation.utils import merge_arguments, construct_kwargs, filter_config_keys, merge_config_and_command_args

logger = logging.getLogger(__name__)

# -----------------------------
# Helper functions for processing and metrics
# -----------------------------
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds, tokenizer, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100 (padding) with the real padding token for decoding.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    raw_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = dict(raw_result)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    # Convert values to rounded numbers or to a string if it is a list.
    result = {
        k: (round(v, 4) if isinstance(v, (float, int))
            else ", ".join(str(x) for x in v) if isinstance(v, list)
            else v)
        for k, v in result.items()
    }
    return result

# -----------------------------
# Main function
# -----------------------------
def main():
    # --- Reading YAML configuration file ---
    # Allows passing the "--config_path" parameter to load arguments from a YAML.
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ExtraArguments, ModelArguments, ProcessorArguments, DataTrainingArguments, ExtendedSeq2SeqTrainingArguments))
    extra_args, model_args, processor_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if extra_args.config_path:
        training_args = merge_config_and_command_args(extra_args.config_path, ExtendedSeq2SeqTrainingArguments, "training", training_args, sys.argv[1:])
        model_args = merge_config_and_command_args(extra_args.config_path, ModelArguments, "model", model_args, sys.argv[1:])
        processor_args = merge_config_and_command_args(extra_args.config_path, ProcessorArguments, "processor", processor_args, sys.argv[1:])
        data_args = merge_config_and_command_args(extra_args.config_path, DataTrainingArguments, "data", data_args, sys.argv[1:])

    # Disable removal of unused columns to ensure correct evaluation.
    setattr(training_args, "remove_unused_columns", False)
    setattr(training_args, "do_predict", True)
    setattr(training_args, "report_to", [])
    setattr(data_args, "visualize_prediction_prob", 0)

    # Send telemetry for usage tracking (optional).
    send_example_telemetry("run_translation", model_args, data_args)

    # --- Logging configuration ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # --- Load the test dataset ---
    if data_args.dataset_dir is not None:
        raw_datasets = load_from_disk(data_args.dataset_dir)
    else:
        raise ValueError("You must specify dataset_dir in the configuration or on the command line.")
    
    if "test" not in raw_datasets:
        raise ValueError("The dataset does not contain a test partition.")
    test_dataset = raw_datasets["test"].with_transform(processor._transform_get_items_output)
    if data_args.max_predict_samples is not None:
        max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
        test_dataset = test_dataset.select(range(max_predict_samples))

    # --- Set seed for reproducibility ---
    set_seed(training_args.seed)

    # --- Load configuration and model ---
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.generation_max_length is not None:
        config.max_new_tokens = training_args.generation_max_length
        config.max_length = None
    elif hasattr(config, "max_new_tokens") and config.max_new_tokens is not None:
        config.max_length = None

    generation_config = GenerationConfig.from_model_config(config)

    # --- Load tokenizer or processor ---
    tokenizer = None
    processor= None
    if not processor_args.processor_name_or_path:
        raise ValueError("You must specify processor_name_or_path in the config or on the command line")
    else:
        processor_kwargs = construct_kwargs(processor_args, ["processor_name_or_path"])
        processor = AutoProcessor.from_pretrained(
            processor_args.processor_name_or_path,
            **processor_kwargs
        )
        for key in set(processor_kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, processor_kwargs.pop(key))
        tokenizer = processor.tokenizer

    # --- Load the pretrained model ---
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # --- Configure the data collator ---
    # Responsible for grouping and preparing data for evaluation; internally manages language aspects.
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if processor is not None:
        data_collator = DataCollatorMultimodalSeq2Seq(
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            label_pad_token_id=label_pad_token_id,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # --- Load the evaluation metric ---
    metric = evaluate.load(training_args.metric_name, cache_dir=model_args.cache_dir)
    training_args.generation_config = generation_config if generation_config is not None else None

    # --- Initialize the Trainer ---
    trainer = MultiLingualSeq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, metric)
            if training_args.predict_with_generate else None,
        visualize_prediction_prob=data_args.visualize_prediction_prob
    )

    logger.info(f"\n{model}\n")
    logger.info(f"\n{print_module_details(model)}\n")

    # --- Execute evaluation ---
    # Predict is invoked to generate predictions and calculate metrics on the test dataset.
    logger.info("*** Evaluation on the test partition ***")
    max_length = training_args.generation_max_length if training_args.generation_max_length is not None else model.max_length
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    predict_results = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams)
    metrics_result = predict_results.metrics
    max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
    metrics_result["predict_samples"] = min(max_predict_samples, len(test_dataset))
    trainer.log_metrics("predict", metrics_result)
    trainer.save_metrics("predict", metrics_result)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            # Retrieve predictions and labels from the predict_results.
            predictions = predict_results.predictions
            label_ids = predict_results.label_ids  # Ensure your dataset provides labels

            # Replace -100 with the tokenizer's pad token id for proper decoding.
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

            # Decode predictions and labels.
            predictions_decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions_decoded = [pred.strip() for pred in predictions_decoded]
            labels_decoded = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            labels_decoded = [lab.strip() for lab in labels_decoded]

            # File to store only the predictions.
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(predictions_decoded))
            logger.info(f"Predictions saved in: {output_prediction_file}")

            # File to store both labels and predictions in the desired format.
            output_full_file = os.path.join(training_args.output_dir, "predictions_labels.txt")
            with open(output_full_file, "w", encoding="utf-8") as writer:
                for idx, (lab, pred) in enumerate(zip(labels_decoded, predictions_decoded)):
                    writer.write(f"L [{idx}] \t{lab}\n")
                    writer.write(f"P [{idx}] \t{pred}\n")
            logger.info(f"Labels and predictions saved in: {output_full_file}")

if __name__ == "__main__":
    main()

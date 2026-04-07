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
import multimodalhugs.processors  # triggers AutoProcessor registration for all processor classes
import multimodalhugs.models
from multimodalhugs import MultiLingualSeq2SeqTrainer


import logging
import os
import sys
import argparse
import warnings

import datasets
import numpy as np
from datasets import load_from_disk

import transformers
from transformers.trainer_utils import get_last_checkpoint

from multimodalhugs.data import DataCollatorMultimodalSeq2Seq
from multimodalhugs.utils import print_module_details

from multimodalhugs.tasks.translation.config_classes import ModelArguments, ProcessorArguments, DataTrainingArguments, ExtraArguments, ExtendedSeq2SeqTrainingArguments, GenerateArguments

from multimodalhugs.tasks.translation.utils import (
    merge_arguments,
    construct_kwargs,
    filter_config_keys,
    merge_config_and_command_args,
    ensure_train_output_dir,
    resolve_missing_arg,
    resolve_checkpoint_path_from_general_setup_path,
)

logger = logging.getLogger(__name__)

# -----------------------------
# Helper functions for processing and metrics
# -----------------------------
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds, tokenizer, metrics_list, metric_names):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100 (padding) with the real padding token for decoding.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = {}
    for metric, name in zip(metrics_list, metric_names):
        raw_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        for k, v in raw_result.items():
            out_key = name if k == "score" else f"{name}_{k}"
            if isinstance(v, (float, int)):
                result[out_key] = round(v, 4)
            elif isinstance(v, list):
                result[out_key] = str(v)
            else:
                result[out_key] = v
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = round(np.mean(prediction_lens), 4)
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
    
    parser = HfArgumentParser((GenerateArguments, ExtraArguments, ModelArguments, ProcessorArguments, DataTrainingArguments, ExtendedSeq2SeqTrainingArguments))
    generate_args, extra_args, model_args, processor_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if extra_args.config_path:
        for section in ("training",):
            try:
                training_args = merge_config_and_command_args(
                    extra_args.config_path,
                    ExtendedSeq2SeqTrainingArguments,
                    section,
                    training_args,
                    sys.argv[1:]
                )
                break
            except KeyError:
                continue
        generate_args = merge_config_and_command_args(extra_args.config_path, GenerateArguments, "generation", generate_args, sys.argv[1:])
        model_args = merge_config_and_command_args(extra_args.config_path, ModelArguments, "model", model_args, sys.argv[1:])
        processor_args = merge_config_and_command_args(extra_args.config_path, ProcessorArguments, "processor", processor_args, sys.argv[1:])
        data_args = merge_config_and_command_args(extra_args.config_path, DataTrainingArguments, "data", data_args, sys.argv[1:])

    # Disable removal of unused columns to ensure correct evaluation.
    setattr(training_args, "remove_unused_columns", False)
    setattr(training_args, "do_predict", True)
    setattr(training_args, "report_to", [])
    setattr(training_args, "visualize_prediction_prob", 0)

    # Apply default manually if user did not provide it
    if generate_args.generate_output_dir is None:
        generate_args.generate_output_dir = os.getcwd()
        logger.warning(f"WARNING: No --generate_output_dir provided. "
            f"Using current directory: {generate_args.generate_output_dir}")
    else:
        logger.info(f"Outputs will be stored in: {generate_args.generate_output_dir}")

    if model_args.model_name_or_path is None:
        resolve_missing_arg(model_args, 'model_name_or_path', training_args.output_dir, extra_args.setup_path if hasattr(extra_args, 'setup_path') else None)
        model_args.model_name_or_path = resolve_checkpoint_path_from_general_setup_path(model_args.model_name_or_path)

    resolve_missing_arg(processor_args, 'processor_name_or_path', training_args.output_dir, extra_args.setup_path if hasattr(extra_args, 'setup_path') else None)
    resolve_missing_arg(data_args, 'dataset_dir', training_args.output_dir, extra_args.setup_path if hasattr(extra_args, 'setup_path') else None)

    # --- Logging configuration ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    _LOG_LEVEL = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
    verbosity = _LOG_LEVEL.get((extra_args.verbosity_level or "warning").lower(), logging.WARNING)

    # Apply unified verbosity to all multimodalhugs loggers (main process only) and HF libraries.
    pkg_level = verbosity if training_args.should_log else logging.WARNING
    logger.setLevel(pkg_level)
    logging.getLogger("multimodalhugs").setLevel(pkg_level)
    datasets.utils.logging.set_verbosity(verbosity)
    transformers.utils.logging.set_verbosity(verbosity)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if verbosity > logging.INFO:
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    # --- Load the test dataset ---
    if data_args.dataset_dir is not None:
        raw_datasets = load_from_disk(data_args.dataset_dir)
    else:
        raise ValueError("You must specify dataset_dir in the configuration or on the command line.")
    


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

    if generate_args.max_length is not None:
        config.max_new_tokens = generate_args.max_length
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

    if "test" not in raw_datasets:
        raise ValueError("The dataset does not contain a test partition.")
    test_dataset = raw_datasets["test"].with_transform(processor._transform_get_items_output)
    if data_args.max_predict_samples is not None:
        max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
        test_dataset = test_dataset.select(range(max_predict_samples))

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
    # `import evaluate` is deferred here and made conditional on `metric_name`
    # being set. `evaluate` transitively imports `transformers.pipelines`
    # (including `video_classification`), which in turn imports `av`. A
    # top-level import would therefore require `av` in every environment just
    # to *load* this module, even in pose-only or text-only setups that never
    # use a metric. Keeping the import conditional means environments without
    # `av` can still run generation as long as no metric is requested.
    metrics_list = []
    metric_names = []
    if training_args.metric_name is not None:
        import evaluate
        metric_names = [m.strip() for m in training_args.metric_name.split(",")]
        metrics_list = [evaluate.load(name, cache_dir=model_args.cache_dir) for name in metric_names]
    training_args.generation_config = generation_config if generation_config is not None else None

    if generate_args.generate_output_dir is not None: # HOTFIX to ensure the trainer stores all_results.json at generate_output_dir directory
        training_args.output_dir = generate_args.generate_output_dir
    # --- Initialize the Trainer ---
    trainer = MultiLingualSeq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            (lambda eval_preds: compute_metrics(eval_preds, tokenizer, metrics_list, metric_names))
            if training_args.predict_with_generate and metrics_list else None
        ),
        visualize_prediction_prob=training_args.visualize_prediction_prob
    )

    logger.info(f"\n{model}\n")
    logger.info(f"\n{print_module_details(model)}\n")

    # --- Execute evaluation ---
    # Predict is invoked to generate predictions and calculate metrics on the test dataset.
    logger.info("*** Evaluation on the test partition ***")
    max_length = generate_args.max_length if generate_args.max_length is not None else model.max_length
    num_beams = generate_args.num_beams

    predict_results = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams)
    metrics_result = predict_results.metrics
    max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
    metrics_result["predict_samples"] = min(max_predict_samples, len(test_dataset))
    trainer.log_metrics("predict", metrics_result)
    trainer.save_metrics("predict", metrics_result, combined=False)

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
            output_prediction_file = os.path.join(generate_args.generate_output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(predictions_decoded))
            logger.info(f"Predictions saved in: {output_prediction_file}")

            # File to store both labels and predictions in the desired format.
            output_full_file = os.path.join(generate_args.generate_output_dir, "predictions_labels.txt")
            with open(output_full_file, "w", encoding="utf-8") as writer:
                for idx, (lab, pred) in enumerate(zip(labels_decoded, predictions_decoded)):
                    writer.write(f"L [{idx}] \t{lab}\n")
                    writer.write(f"P [{idx}] \t{pred}\n")
            logger.info(f"Labels and predictions saved in: {output_full_file}")

if __name__ == "__main__":
    main()

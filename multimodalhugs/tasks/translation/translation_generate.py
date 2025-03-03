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

import logging
import os
import sys
import dataclasses
import argparse
from omegaconf import OmegaConf
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Optional, List, TypeVar

import datasets
import evaluate
import numpy as np
from datasets import load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoProcessor,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    GenerationConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

# Import multimodal framework components: processors, models, and trainer.
from multimodalhugs.processors import SignwritingProcessor, Pose2TextTranslationProcessor, Image2TextTranslationProcessor
from multimodalhugs.models import MultiModalEmbedderModel, MultiModalEmbedderConfig
from multimodalhugs import MultiLingualSeq2SeqTrainer
from multimodalhugs.data import DataCollatorMultimodalSeq2Seq
from multimodalhugs.utils import print_module_details

# Register custom configurations and classes for our models and processors.
AutoConfig.register("multimodal_embedder", MultiModalEmbedderConfig)
AutoModelForSeq2SeqLM.register(MultiModalEmbedderConfig, MultiModalEmbedderModel)

Pose2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("pose2text_translation_processor", Pose2TextTranslationProcessor)

SignwritingProcessor.register_for_auto_class()
AutoProcessor.register("signwritting_processor", SignwritingProcessor)

Image2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("image2text_translation_processor", Image2TextTranslationProcessor)

logger = logging.getLogger(__name__)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

T = TypeVar("T")

def construct_kwargs(obj, not_used_keys=[]):
    kwargs = {}
    obj_dict = asdict(obj)

    for field_info in fields(obj):
        if field_info.name in not_used_keys:
            continue
        # Check if the field has a default factory
        if field_info.default_factory is not dataclasses.MISSING:  # Handle fields with default_factory
            default_value = field_info.default_factory()
        else:
            default_value = field_info.default
        # Compare current value with default value
        if obj_dict[field_info.name] != default_value:
            kwargs[field_info.name] = obj_dict[field_info.name]
    
    return kwargs

# -----------------------------
# Functions for managing YAML configuration
# -----------------------------
def merge_arguments(cmd_args: T, extra_args: T, command_arg_names: List[str], yaml_arg_keys: List[str]) -> T:
    """
    Merges command line arguments with those defined in the YAML.
    Values that were not explicitly specified on the command line are overwritten.
    """
    if not (is_dataclass(cmd_args) and is_dataclass(extra_args)):
        raise ValueError("Both cmd_args and extra_args must be dataclass instances.")
    
    default_arguments = [f.name for f in fields(cmd_args) if f.name not in command_arg_names]
    for f in fields(cmd_args):
        if f.name in yaml_arg_keys:
            cmd_value = getattr(cmd_args, f.name)
            cfg_value = getattr(extra_args, f.name)
            if cmd_value != cfg_value and f.name in default_arguments:
                setattr(cmd_args, f.name, cfg_value)
    return cmd_args

def filter_config_keys(config_section: dict, dataclass_type) -> dict:
    """Filters the valid keys for the dataclass."""
    valid_keys = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in config_section.items() if k in valid_keys}

def merge_config_and_command_args(config_path, class_type, section, _args, remaining_args):
    """
    Merges the arguments from the YAML file with the command line arguments for a given section.
    """
    yaml_conf = OmegaConf.load(config_path)
    yaml_dict = OmegaConf.to_container(yaml_conf, resolve=True)
    _parser = HfArgumentParser((class_type,))
    filtered_yaml = filter_config_keys(yaml_dict[section], class_type)
    extra_args = _parser.parse_dict(filtered_yaml)[0]
    command_arg_names = [value[2:].replace("-", "_") for value in remaining_args if value[:2] == '--']
    yaml_keys = yaml_dict[section].keys()
    _args = merge_arguments(
        cmd_args=_args,
        extra_args=extra_args,
        command_arg_names=command_arg_names,
        yaml_arg_keys=yaml_keys
    )
    return _args

# -----------------------------
# Definition of arguments
# -----------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to the pretrained model or identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of the pretrained configuration (if different from model_name)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of the pretrained tokenizer (if different from model_name)"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store downloaded models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific version of the model (branch, tag, or commit)"}
    )
    token: str = field(
        default=None,
        metadata={"help": "Token for downloading remote files"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote model code execution"}
    )

@dataclass
class ProcessorArguments:
    processor_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pretrained processor or identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    dataset_dir: Optional[str] = field(default=None, metadata={"help": "Data directory"})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name (using the datasets library)"}
    )  # Not implemented yet
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset configuration"}
    )  # Not implemented yet
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Test data file for evaluation (jsonlines)"}
    )  # Not implemented yet
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the data cache"}
    )  # Not implemented yet
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for preprocessing"}
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum input sequence length"}
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum target sequence length"}
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length for evaluation (uses max_target_length if not defined)"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad to the maximum length"}
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to evaluate"}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for generation"}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Ignore padding tokens in the loss"}
    )
    # Although defined, these arguments are no longer used directly in generate.
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "Prefix for the input (not used in generate)"}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token forced as the first token in generation (for multilingual models)"}
    )
    metric_name: Optional[str] = field(
        default="sacrebleu",
        metadata={"help": "Name of the metric to use (any metric supported by evaluate.load())"}
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

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
    # Allows passing the "--config-path" parameter to load arguments from a YAML.
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--config-path", type=str, help="Path to YAML config file")
    extra_parser.add_argument("--visualize_prediction_prob", type=float, default=0.0, help="Percentage of samples displaying their predictions during evaluation")
    extra_args, remaining_args = extra_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args  # Remove configuration arguments for the next parser

    # --- Parse arguments into dataclasses ---
    parser = HfArgumentParser((ModelArguments, ProcessorArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, processor_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if extra_args.config_path:
        training_args = merge_config_and_command_args(extra_args.config_path, Seq2SeqTrainingArguments, "training", training_args, remaining_args)
        model_args = merge_config_and_command_args(extra_args.config_path, ModelArguments, "model", model_args, remaining_args)
        processor_args = merge_config_and_command_args(extra_args.config_path, ProcessorArguments, "processor", processor_args, remaining_args)
        data_args = merge_config_and_command_args(extra_args.config_path, DataTrainingArguments, "data", data_args, remaining_args)

    # Disable removal of unused columns to ensure correct evaluation.
    setattr(training_args, "remove_unused_columns", False)

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
    test_dataset = raw_datasets["test"]
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
    processor = None
    if not processor_args.processor_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
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

    # --- Preprocess the test dataset ---
    # Tokenize the 'translation' column in a generic way, without manually managing source_lang/target_lang or prefixes.
    padding = "max_length" if data_args.pad_to_max_length else False
    max_target_length = data_args.val_max_target_length

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
    elif data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # --- Load the evaluation metric ---
    metric = evaluate.load(data_args.metric_name, cache_dir=model_args.cache_dir)
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
        visualize_prediction_prob=extra_args.visualize_prediction_prob
    )

    logger.info(f"\n{model}\n")
    logger.info(f"\n{print_module_details(model)}\n")

    # --- Execute evaluation ---
    # Predict is invoked to generate predictions and calculate metrics on the test dataset.
    logger.info("*** Evaluation on the test partition ***")
    max_length = training_args.generation_max_length if training_args.generation_max_length is not None else data_args.val_max_target_length
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

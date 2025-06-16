#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for sequence to sequence.
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
    EarlyStoppingCallback,
)

from multimodalhugs.processors import (
    SignwritingProcessor,
    Pose2TextTranslationProcessor,
    Video2TextTranslationProcessor,
    Image2TextTranslationProcessor,
    Text2TextTranslationProcessor,
    Features2TextTranslationProcessor
)

import multimodalhugs.models
from multimodalhugs import MultiLingualSeq2SeqTrainer

Pose2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("pose2text_translation_processor", Pose2TextTranslationProcessor)

Video2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("video2text_translation_processor", Video2TextTranslationProcessor)

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
from multimodalhugs.tasks.translation.utils import merge_arguments, construct_kwargs, filter_config_keys, merge_config_and_command_args, check_t5_fp16_compatibility

logger = logging.getLogger(__name__)

def main():
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
            
    # set remove_unused_columns to false
    setattr(training_args, "remove_unused_columns", False)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_dir is not None:
        raw_datasets = load_from_disk(
            data_args.dataset_dir,
        )
    else:
        raise ValueError("You must specify dataset_dir in the config or on the command line")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if hasattr(config, "max_new_tokens") and config.max_new_tokens is not None:
        # Avoids the warning about setting a value for both max_length and max_new_tokens
        config.max_length = None


    generation_config = GenerationConfig.from_model_config(config)


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

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    check_t5_fp16_compatibility(model, training_args.fp16)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Check the whether the source target length fits in the model, if it has absolute positional embeddings
    if (
        hasattr(model.config, "max_position_embeddings")
        and not hasattr(model.config, "relative_attention_max_distance")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        raise ValueError(
            f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
            f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
            f" `--max_source_length` to {model.config.max_position_embeddings} or using a model with larger position "
            "embeddings"
        )


    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"].with_transform(processor._transform_get_items_output)
        print(f"train_dataset: {train_dataset}")
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"].with_transform(processor._transform_get_items_output)
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"].with_transform(processor._transform_get_items_output)
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))


    # Data collator
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
    
    # Load metric(s)
    if training_args.metric_name is not None:
        metric_names = [m.strip() for m in training_args.metric_name.split(",")]
        metrics_list = [evaluate.load(name, cache_dir=model_args.cache_dir) for name in metric_names]

        # Check that metric_for_best_model is among the loaded metric_names
        if training_args.metric_for_best_model is not None and training_args.metric_for_best_model not in metric_names:
            raise ValueError(
                f"You specified 'metric_for_best_model={training_args.metric_for_best_model}', "
                f"but it's not among the listed 'metric_name' values ({metric_names}). "
                f"If you want to track a specific metric for selecting the best model, make sure it's also included in 'metric_name'."
            )
    else:
        metrics_list = []

    # Define torch_empty_cache_steps to optimize memory utilization
    training_args.torch_empty_cache_steps = training_args.gradient_accumulation_steps

    # Add the generation_config in case generation is performed during training
    training_args.generation_config = generation_config if generation_config is not None else None

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        if not metrics_list:
            return {}

        compute_metrics_tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding
        preds = np.where(preds != -100, preds, compute_metrics_tokenizer.pad_token_id)
        decoded_preds = compute_metrics_tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, compute_metrics_tokenizer.pad_token_id)
        decoded_labels = compute_metrics_tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = {}
        for metric, name in zip(metrics_list, metric_names):
            metric_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result[name] = round(metric_result.get("score", metric_result.get(name, 0.0)), 4)

        prediction_lens = [np.count_nonzero(pred != compute_metrics_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = round(np.mean(prediction_lens), 4)
        return result

    callbacks_list = []
    if training_args.early_stopping_patience is not None:
        callbacks_list.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    # Initialize our Trainer
    trainer = MultiLingualSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        visualize_prediction_prob=training_args.visualize_prediction_prob,
        print_decoder_prompt_on_prediction=training_args.print_decoder_prompt_on_prediction,
        print_special_tokens_on_prediction=training_args.print_special_tokens_on_prediction,
        callbacks=callbacks_list
    )

    logger.info(f"\n{model}\n")
    logger.info(f"\n{print_module_details(model)}\n")

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info(f"Resuming training from: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    metrics_result = {}
    if training_args.do_predict:
        logger.info("*** Evaluation on the test partition ***")
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else model.max_length
        )
        num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams)
        metrics_result = predict_results.metrics
        print(metrics_result)
    return metrics_result

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
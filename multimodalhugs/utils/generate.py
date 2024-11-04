import os
import torch
import itertools
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from evaluate import load
from collections import defaultdict
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from datasets import load_dataset, load_from_disk
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Union

from multimodalhugs.data import DataCollatorMultimodalSeq2Seq
from multimodalhugs.models import MultiModalEmbedderModel, MultiModalEmbedderConfig
from multimodalhugs.processors import SignwritingProcessor, Pose2TextTranslationProcessor, Image2TextTranslationProcessor

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoProcessor,
    GenerationConfig,
)

### Register

AutoConfig.register("multimodal_embedder", MultiModalEmbedderConfig)
AutoModelForSeq2SeqLM.register(MultiModalEmbedderConfig, MultiModalEmbedderModel)

Pose2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("pose2text_translation_processor", Pose2TextTranslationProcessor)

SignwritingProcessor.register_for_auto_class()
AutoProcessor.register("signwritting_processor", SignwritingProcessor)

Image2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("image2text_translation_processor", Image2TextTranslationProcessor)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    checkpoint_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    processor_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )

@dataclass
class DataEvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_dir: Optional[str] = field(default=None, metadata={"help": "Path to the data directory"})
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to the evaluation and generation process.
    """
    metric_name: str = field(
        default="bleu",
        metadata={
            "help": "Metric identifier. See https://huggingface.co/docs/evaluate/en/index for available metrics."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated; sequences shorter will be padded."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams for evaluation. This argument will be passed to `model.generate`, "
                "used during `evaluate` and `predict`."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use fp16 (mixed) precision instead of 32-bit precision."
        },
    )
    batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size per device for evaluation."
        },
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory where the evaluation results will be stored."
        },
    )

### utils

def run_evaluation(model, tokenizer, dataloader, device, evaluation_args, generation_config, metric):
    """
    Evaluates the model using the provided dataloader and computes the specified metric.
    Supports mixed-precision computation if fp16 is enabled and a CUDA device is available.

    Args:
        model (nn.Module): The model to evaluate.
        tokenizer (PreTrainedTokenizerBase): The tokenizer associated with the model.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to run the evaluation on.
        data_args (DataEvaluationArguments): Evaluation arguments containing configurations.
        generation_config (GenerationConfig): Generation configurations for the model.
        metric: The evaluation metric to compute.

    Returns:
        Tuple containing lists of decoded texts and the evaluation results.
    """
    T_list = []
    P_list = []
    L_list = []
    H_list = []

    model.eval()

    # Determine whether to use autocast based on fp16 and device type
    if evaluation_args.fp16 and device.type == 'cuda':
        from torch.cuda.amp import autocast
    else:
        # Define a dummy context manager for autocast when not using fp16
        from contextlib import contextmanager

        @contextmanager
        def autocast():
            yield

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move tensors to the appropriate device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast():
                generated_tokens, labels = generation_step(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=batch,
                    **generation_config.__dict__,
                )

            # Decode the texts
            T, P, L, H = get_decoded_texts(tokenizer=tokenizer, preds=generated_tokens, labels=labels)

            # Extend the lists with the decoded texts
            T_list.extend(T)
            P_list.extend(P)
            L_list.extend(L)
            H_list.extend(H)

    # Compute the evaluation metric
    results = metric.compute(predictions=P_list, references=H_list)

    return T_list, P_list, L_list, H_list, results


def generation_step(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    **gen_kwargs,
):
    
    def _pad_tensors_to_max_len(tensor, max_length):
        if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        else:
            if model.config.pad_token_id is not None:
                pad_token_id = model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    generation_inputs = inputs.copy()
    # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
    # (otherwise, it would continue generating from the padded `decoder_input_ids`)
    if (
        "labels" in generation_inputs
        and "decoder_input_ids" in generation_inputs
        and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
    ):
        generation_inputs = {
            k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
        }
    labels = inputs['labels']
    lang_tokens = inputs['labels'][:, 0].tolist()
    # Use these language tokens to group indices
    language_groups = defaultdict(list)
    for idx, lang_token in enumerate(lang_tokens):
        language_groups[lang_token].append(idx)

    generated_tokens = []
    original_indices = []
    max_len_generation = 0

    # Generamos para cada miniminibatch agrupado por idioma

    for lang_token, sample_indices in language_groups.items():
        miniminibatch_inputs = {k: v[sample_indices] for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")}

        gen_kwargs["forced_bos_token_id"] = int(lang_token)

        # Generamos para este miniminibatch
        batch_generated_tokens = model.generate(**miniminibatch_inputs, **gen_kwargs)
        original_indices.extend(sample_indices)


        if batch_generated_tokens.shape[1] > max_len_generation:
            max_len_generation = batch_generated_tokens.shape[1]
        
        generated_tokens.append(batch_generated_tokens)

    sorted_indices = sorted(range(len(original_indices)), key=lambda k: original_indices[k])

    for i in range(len(generated_tokens)):
        generated_tokens[i] = F.pad(generated_tokens[i], (0, max_len_generation - generated_tokens[i].size(1)), value=tokenizer.pad_token_id)

    generated_tokens = torch.cat(generated_tokens, dim=0)
    generated_tokens = generated_tokens[sorted_indices]

    # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
    # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
    # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
    if model.generation_config._from_model_config:
        model.generation_config._from_model_config = False

    # in case the batch is shorter than max length, the output should be padded
    if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
        generated_tokens = _pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

    return generated_tokens, labels

def get_dict_item_by_position(d, position):
    return next(itertools.islice(d.items(), position, None))

def get_decoded_texts(tokenizer, preds, labels):
    labels_with_special_tokens = tokenizer.batch_decode(labels, skip_special_tokens=False)
    preds_with_special_tokens = tokenizer.batch_decode(preds, skip_special_tokens=False)
    labels_without_special_tokens = tokenizer.batch_decode(labels, skip_special_tokens=True)
    preds_without_special_tokens = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return labels_with_special_tokens, preds_with_special_tokens, labels_without_special_tokens, preds_without_special_tokens


## code

def main():
    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, EvaluationArguments))
    model_args, data_args, evaluation_args = parser.parse_args_into_dataclasses()

    # Output the parsed arguments
    print(f"Model Args: \n{model_args}\n")
    print(f"Data Args: \n{data_args}\n")
    print(f"Evaluation Args: \n{evaluation_args}\n")

    #### Modules

    # Load the configuration

    # Load the processor and tokenizer
    if model_args.processor_name_or_path:
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        processor = None  # Ensure 'processor' is defined
        
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    data_collator = DataCollatorMultimodalSeq2Seq(
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        )
    
    num_workers = data_args.preprocessing_num_workers if data_args.preprocessing_num_workers is not None else len(os.sched_getaffinity(0))

    #### Data
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif data_args.dataset_dir is not None:
        raw_datasets = load_from_disk(
            data_args.dataset_dir,
        )
    else:
        data_files = {}
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        if extension == "jsonl":
            builder_name = "json"  # the "json" builder reads both .json and .jsonl files
        else:
            builder_name = extension  # e.g. "parquet"
        raw_datasets = load_dataset(
            builder_name,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        
    if "test" not in raw_datasets:
        raise ValueError("Evaluating the model requires a test dataset")
    test_dataset = raw_datasets["test"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_params = {
        "batch_size": evaluation_args.batch_size,
        "collate_fn": data_collator,
        "num_workers": num_workers,
        "pin_memory": True if torch.cuda.is_available() else False,
    }

    dataloader = DataLoader(test_dataset, **dataloader_params)

    if evaluation_args.num_beams != 1:
        model.generation_config.num_beams = evaluation_args.num_beams

    metric = load(evaluation_args.metric_name)

    model.to(device)
    model.eval()

    T_list, P_list, L_list, H_list, results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        evaluation_args=evaluation_args,
        generation_config=model.generation_config,
        metric=metric,
    )

    metric_name, value = get_dict_item_by_position(results, 0)
    results.pop(metric_name)
    last_line_string = f"{metric_name.upper()} Score: {value}; Metric details: {results}"

    output_file = evaluation_args.output_path + '/evaluation_results.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(T_list)):
            f.write(f"\nT[{i}] - {T_list[i]}\n")
            f.write(f"L[{i}] - {L_list[i]}\n")
            f.write(f"P[{i}] - {P_list[i]}\n")
            f.write(f"H[{i}] - {H_list[i]}\n")
        f.write(f"\n{last_line_string}")

if __name__ == "__main__":
    main()
import logging

from dataclasses import dataclass, field
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
                "Whether to trust the execution of code from datasets/models defined on the Hub. "
                "This option should only be set to `True` for repositories you trust and in which you have read the "
                "code, as it will execute code present on the Hub on your local machine."
            )
        },
    )

@dataclass
class ProcessorArguments:
    processor_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_dir: Optional[str] = field(default=None, metadata={"help": "Path to the data directory"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

@dataclass
class ExtraArguments:
    """
    Allows passing the "--config_path" parameter to load arguments from a YAML.
    """
    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML config file"}
    )
    setup_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the setup directory containing actors_paths.yaml, used only if model_name_or_path, processor_name_or_path, or dataset_dir are not provided via commandline or config file. In train, if not provided, the path is inferred from training_args.output_dir/setup."}
    )

@dataclass
class ExtendedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    metric_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the metric to use (any metric supported by evaluate.load()). If you want to use multiple metrics, structure the variable like: metric_name: '<metric_name_1>,<metric_name_2>,...'"}
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Use with `EarlyStoppingCallback`. "
                "Number of evaluation calls with no improvement after which training will be stopped. "
                "Requires `load_best_model_at_end=True`, `metric_for_best_model` set, and a valid `evaluation_strategy`. "
                "If set to `None`, early stopping is disabled."
            )
        }
    )
    visualize_prediction_prob: float = field(
        default=0.05,
        metadata={"help": "Percentage of samples displaying their predictions during evaluation"}
    )
    print_decoder_prompt_on_prediction: bool = field(
        default=False,
        metadata={
            "help": "If True, adds a field called 'Prompt' when performing prediction."
        },
    )
    print_special_tokens_on_prediction: bool = field(
        default=False,
        metadata={
            "help": "If True, also prints a version of prediction and ground truth that contain special tokens."
        },
    )
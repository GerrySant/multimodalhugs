import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import torch
import random
from torch import nn
from torch.utils.data import Dataset

import torch.nn.functional as F

from transformers import Trainer, Seq2SeqTrainer

from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction, PredictionOutput
    from transformers.training_args import TrainingArguments

def all_values_equal(tensor):
    if tensor.numel() == 0:  # Check if the tensor is empty, thus, no generation_prompt specified.
        return False
    return torch.all(tensor == tensor.flatten()[0])

class MultiLingualSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        visualize_prediction_prob: float = 0.05
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config
        self.visualize_prediction_prob = visualize_prediction_prob

    def visualize_generation(self, preds, labels):

        pad_id = self.tokenizer.pad_token_id
        labels[labels == -100] = pad_id

        T = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        P = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        L = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        H = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        for i in range(len(T)):
            print(f"\nT - {T[i]}")
            print(f"L - {L[i]}")
            print(f"P - {P[i]}")
            print(f"H - {H[i]}")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        gen_kwargs = self.model.generation_config.to_dict()
        if not self.args.predict_with_generate or prediction_loss_only:
            return Trainer.prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )
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

        if all_values_equal(generation_inputs['decoder_attention_mask']):
            # If all decoder_prompts have the same number of tokens, we can pass the whole batch in the model.generate()
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        elif generation_inputs['decoder_attention_mask'].numel() == 0:
            # If decoder_prompts are empty, remove the empty tensors from generation_inputs before calling model.generate()
            generation_inputs.pop("decoder_input_ids", None)
            generation_inputs.pop("decoder_attention_mask", None)
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        else:
            # Otherwise, we generate sample by sample:
            B = next(iter(generation_inputs.values())).shape[0]
            samples = [{key: value[i:i+1] for key, value in generation_inputs.items()} for i in range(B)]

            generated_tokens = []
            max_len_generation = 0

            for sample in samples:
                _generated_tokens = self.model.generate(**sample, **gen_kwargs)

                if _generated_tokens.shape[1] > max_len_generation:
                    max_len_generation = _generated_tokens.shape[1]

                generated_tokens.append(_generated_tokens)

            for i in range(len(generated_tokens)):
                generated_tokens[i] = F.pad(generated_tokens[i], (0, max_len_generation - generated_tokens[i].size(1)), value=self.tokenizer.pad_token_id)
            generated_tokens = torch.cat(generated_tokens, dim=0)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False
            
        self.generation_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if hasattr(self, "generation_max_length") and self.generation_max_length is not None and generated_tokens.shape[-1] < self.generation_max_length + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.generation_max_length + 1)
        elif self.generation_config.max_new_tokens is not None: 
            if generated_tokens.shape[-1] < self.generation_config.max_new_tokens + 1:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.generation_config.max_new_tokens + 1)
        elif generated_tokens.shape[-1] < self.generation_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.generation_config.max_length)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if hasattr(self, "generation_max_length") and self.generation_max_length is not None and generated_tokens.shape[-1] < self.generation_max_length + 1:
                labels = self._pad_tensors_to_max_len(labels, self.generation_max_length + 1)
            elif self.generation_config.max_new_tokens is not None:
                if labels.shape[-1] < self.generation_config.max_new_tokens + 1:
                    labels = self._pad_tensors_to_max_len(labels, self.generation_config.max_new_tokens + 1)
            elif labels.shape[-1] < self.generation_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, self.generation_config.max_length)
        else:
            labels = None
        if random.random() < self.visualize_prediction_prob:
            self.visualize_generation(preds=generated_tokens, labels=labels)
        return loss, generated_tokens, labels
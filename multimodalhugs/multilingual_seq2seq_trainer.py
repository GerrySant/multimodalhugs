import contextlib
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
from transformers.integrations.fsdp import is_fsdp_managed_module

if torch.distributed.is_available():
    from torch.distributed.fsdp import FullyShardedDataParallel
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction, PredictionOutput
    from transformers.training_args import TrainingArguments

def all_values_equal(tensor):
    if tensor.numel() == 0:  # Check if the tensor is empty, thus, no decoder_prompt specified.
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
        processing_class: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        visualize_prediction_prob: float = 0.05,
        print_decoder_prompt_on_prediction: bool = False,
        print_special_tokens_on_prediction: bool = False

    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
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
        self.print_decoder_prompt_on_prediction = print_decoder_prompt_on_prediction
        self.print_special_tokens_on_prediction = print_special_tokens_on_prediction

    def visualize_generation(self, preds, labels):

        pad_id = self.processing_class.pad_token_id
        labels[labels == -100] = pad_id

        decoded_label_with_special_tokens = self.processing_class.batch_decode(labels, skip_special_tokens=False)
        decoded_prediction_with_special_tokens = self.processing_class.batch_decode(preds, skip_special_tokens=False)
        decoded_label = self.processing_class.batch_decode(labels, skip_special_tokens=True)
        decoded_prediction = self.processing_class.batch_decode(preds, skip_special_tokens=True)

        for i in range(len(decoded_label_with_special_tokens)):
            print("")
            if self.print_special_tokens_on_prediction:
                print(f"Label with special tokens - {decoded_label_with_special_tokens[i]}")
            print(f"Label - {decoded_label[i]}")
            if self.print_decoder_prompt_on_prediction:
                print(f"Decoder prompt - {decoded_prediction_with_special_tokens[i].split(decoded_prediction[i])[0]}")
            if self.print_special_tokens_on_prediction:
                print(f"Prediction with special tokens - {decoded_prediction_with_special_tokens[i]}")
            print(f"Prediction - {decoded_prediction[i]}")
            

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
        if not self.args.predict_with_generate or prediction_loss_only:
            return Trainer.prediction_step(
                self, model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
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

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)
        generation_inputs = inputs.copy()

        # If decoder_input_ids was created from labels (shifted labels), evict it so generate()
        # starts from the correct prefix.  After the slot rename, the raw decoder prompt is in
        # decoder_prompt_ids; decoder_input_ids is always the teacher-forcing sequence.
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        # Extract the per-sample generation prefix (renamed from decoder_input_ids slot).
        decoder_prompt_ids  = generation_inputs.pop("decoder_prompt_ids", None)
        decoder_prompt_mask = generation_inputs.pop("decoder_prompt_mask", None)

        _bos = self.model.config.decoder_start_token_id
        if _bos is None:
            _bos = self.model.generation_config.decoder_start_token_id
        bos = _bos

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if torch.distributed.is_available() and isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            if decoder_prompt_ids is None or (
                decoder_prompt_mask is not None and decoder_prompt_mask.numel() == 0
            ):
                # No decoder prompt (T5, ByT5, BART, mBART-cc25 with empty decoder_prompt).
                # generate() starts from model.config.decoder_start_token_id automatically.
                generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

            elif all_values_equal(decoder_prompt_mask):
                # All samples have the same prompt — batched generation.
                # Build prefix: [bos, prompt_tok_1, ...] shape [B, 1+P]
                B = decoder_prompt_ids.shape[0]
                bos_col  = decoder_prompt_ids.new_full((B, 1), bos)
                bos_mask = torch.ones(B, 1, dtype=decoder_prompt_mask.dtype,
                                      device=decoder_prompt_mask.device)
                generation_inputs["decoder_input_ids"]      = torch.cat([bos_col, decoder_prompt_ids], dim=1)
                generation_inputs["decoder_attention_mask"] = torch.cat([bos_mask, decoder_prompt_mask], dim=1)
                generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

            else:
                # Different prompt lengths per sample — generate one by one.
                B = decoder_prompt_ids.shape[0]
                generated_tokens_list = []
                max_len = 0
                for i in range(B):
                    actual_len    = int(decoder_prompt_mask[i].sum().item())
                    actual_prompt = decoder_prompt_ids[i, :actual_len]
                    bos_t         = actual_prompt.new_full((1,), bos)
                    prefix        = torch.cat([bos_t, actual_prompt]).unsqueeze(0)
                    prefix_mask   = torch.ones_like(prefix)
                    sample_inputs = {
                        **{k: v[i:i+1] for k, v in generation_inputs.items()},
                        "decoder_input_ids":      prefix,
                        "decoder_attention_mask": prefix_mask,
                    }
                    out = self.model.generate(**sample_inputs, **gen_kwargs)
                    if out.shape[1] > max_len:
                        max_len = out.shape[1]
                    generated_tokens_list.append(out)
                generated_tokens = torch.cat(
                    [F.pad(t, (0, max_len - t.size(1)),
                           value=self.processing_class.pad_token_id)
                     for t in generated_tokens_list],
                    dim=0,
                )

        # Temporary hack to ensure the generation config is not initialized for each iteration of the
        # evaluation loop. Matches the upstream Seq2SeqTrainer pattern added in transformers 5.x.
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Fill in any None fields on generation_config with their defaults so that
        # comparisons like `shape[-1] < gen_config.max_length` never raise TypeError.
        gen_config = self.model.generation_config
        gen_config.update(**gen_config._get_default_generation_params(), defaults_only=True)
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).detach().mean()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).detach().mean()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None
        if random.random() < self.visualize_prediction_prob:
            self.visualize_generation(preds=generated_tokens, labels=labels)
        return loss, generated_tokens, labels
# Standard libraries
import logging

# Third-party libraries
import torch
import torch.nn as nn

# Other
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def freeze_module_parameters(module):
    """
    Freeze the parameters of the provided module to prevent them from being updated during training.
    Logs the action of freezing the module.

    Args:
    module (torch.nn.Module): The module whose parameters are to be frozen.
    """
    for param in module.parameters():
        param.requires_grad = False
    logger.info(f"The parameters of the module {module.__class__.__name__} have been frozen.")

class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_frames: Optional[torch.LongTensor] = None,
        src_prompt: Optional[torch.LongTensor] = None,
        source_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
    ):
        return self.model.input_to_encoder_outputs(
            input_frames = input_frames,
            src_prompt = src_prompt,
            source_prompt_length_padding_mask = source_prompt_length_padding_mask,
            input_ids = input_ids,
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
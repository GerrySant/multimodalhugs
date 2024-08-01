import torch
import logging

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
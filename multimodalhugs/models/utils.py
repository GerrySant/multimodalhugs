# Standard libraries
import logging
import importlib
import inspect

# Third-party libraries
import torch
import torch.nn as nn

from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_WITH_LM_HEAD_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES

# Other
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_encoder_wrapper(model):
    """
    Automatically creates an encoder wrapper that delegates calls to
    `model.input_to_encoder_outputs(...)`, preserving the same signature.

    This ensures compatibility with Hugging Face's generation code, which
    expects the following pattern:

        encoder = model.get_encoder()
        encoder_outputs = encoder(**inputs)

    ⚙️ **Purpose:**
    This utility is designed for multimodal models that **do not define an encoder
    explicitly**, unlike standard Hugging Face models (e.g., BART, T5, M2M100) that 
    already expose an `encoder` submodule. 

    Instead, these models define a method called `input_to_encoder_outputs()`, which
    is responsible for processing multimodal inputs (such as images, poses, or other
    non-text modalities) and returning encoder representations compatible with
    text generation pipelines.

    The dynamically generated wrapper class created here allows Hugging Face’s 
    generation utilities to work seamlessly with such models by mimicking the 
    expected encoder interface.

    Example:
        >>> model = MultiModalEmbedderModel(config)
        >>> encoder = model.get_encoder()
        >>> encoder_outputs = encoder(input_frames=frames, encoder_prompt=prompt)

    Parameters
    ----------
    model : nn.Module
        The multimodal model defining an `input_to_encoder_outputs()` method.

    Returns
    -------
    nn.Module
        A dynamically generated encoder wrapper compatible with the Hugging Face
        generation API.
    """
    # --- Safety check ---
    if not hasattr(model, "input_to_encoder_outputs") or not callable(model.input_to_encoder_outputs):
        raise AttributeError(
            f"[MultimodalHugs Error] The model '{model.__class__.__name__}' does not define a method "
            f"'input_to_encoder_outputs()'.\n\n"
            "Encoder-decoder models must provide a way to obtain encoder representations. "
            "This can be achieved in one of two ways:\n"
            "  1. Define an explicit encoder module as `self.encoder`, similar to Hugging Face models like BART or T5.\n"
            "  2. Implement a method named `input_to_encoder_outputs(self, **kwargs)` that returns the encoder outputs "
            "given multimodal inputs.\n\n"
            "The `input_to_encoder_outputs()` method is required for models that integrate non-text modalities "
            "and do not have a predefined `self.encoder`.\n\n"
            "Example implementation:\n"
            "    def input_to_encoder_outputs(self, **kwargs):\n"
            "        # Process multimodal inputs and return encoder outputs compatible with Seq2Seq models\n"
            "        return self.backbone.encoder(**kwargs)\n"
        )
        
    sig = inspect.signature(model.input_to_encoder_outputs)

    class AutoEncoderWrapper(nn.Module):
        def __init__(self, model_ref):
            super().__init__()
            self.model = model_ref

        def forward(self, *args, **kwargs):
            # Match only valid parameters from the model.input_to_encoder_outputs signature
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in sig.parameters
            }
            return self.model.input_to_encoder_outputs(*args, **filtered_kwargs)

    AutoEncoderWrapper.__name__ = f"{model.__class__.__name__}EncoderWrapper"
    return AutoEncoderWrapper(model)

def get_backbone_config_class(model_type: str):
    """
    Retrieves the specific configuration class for the given `model_type`
    using Hugging Face's CONFIG_MAPPING_NAMES mapping.

    Args:
        model_type (str): The model type (e.g., 'bert', 't5', etc.).

    Returns:
        The corresponding configuration class.

    Raises:
        ValueError: If `model_type` is not found in CONFIG_MAPPING_NAMES.
        ImportError: If the configuration class cannot be imported.
    """
    if model_type not in CONFIG_MAPPING_NAMES:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available options: {list(CONFIG_MAPPING_NAMES.keys())}"
        )

    config_class_name = CONFIG_MAPPING_NAMES[model_type]
    # Assumes that the module is named using model_type, replacing dashes with underscores
    module_name = model_type.replace("-", "_")

    try:
        module = importlib.import_module(f"transformers.models.{module_name}")
    except ImportError:
        # If it fails, try retrieving the class directly from the transformers package.
        import transformers
        if hasattr(transformers, config_class_name):
            return getattr(transformers, config_class_name)
        raise ImportError(f"Could not import module for model_type '{model_type}'.")

    if hasattr(module, config_class_name):
        return getattr(module, config_class_name)
    else:
        # Fallback: Try retrieving the class from the transformers package.
        import transformers
        if hasattr(transformers, config_class_name):
            return getattr(transformers, config_class_name)
    
    raise ImportError(
        f"The configuration class '{config_class_name}' was not found for model_type '{model_type}'."
    )


def get_backbone_model_class(model_type: str):
    """
    Retrieves the specific model (backbone) class for the given `model_type`
    using Hugging Face's MODEL_WITH_LM_HEAD_MAPPING_NAMES and 
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES mappings.

    Args:
        model_type (str): The model type (e.g., 'bert', 't5', etc.).

    Returns:
        The corresponding model class.

    Raises:
        ValueError: If `model_type` is not found in either mapping.
        ImportError: If the module or model class cannot be imported.
    """
    model_class_name = None
    
    if model_type in MODEL_WITH_LM_HEAD_MAPPING_NAMES:
        model_class_name = MODEL_WITH_LM_HEAD_MAPPING_NAMES[model_type]

    if model_type in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
        # If the model appears in both, you could log a warning or choose a default behavior.
        if model_class_name:
            logger.info(
                f"Model type '{model_type}' found in both mappings. Using MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES."
            )
        model_class_name = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES[model_type]

    if not model_class_name:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available options: "
            f"{list(MODEL_WITH_LM_HEAD_MAPPING_NAMES.keys())} and "
            f"{list(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.keys())}"
        )

    module_name = model_type.replace("-", "_")
    
    try:
        module = importlib.import_module(f"transformers.models.{module_name}")
    except ImportError:
        # Fallback: Try retrieving the class from the transformers package
        import transformers
        if hasattr(transformers, model_class_name):
            return getattr(transformers, model_class_name)
        raise ImportError(f"Could not import module for model_type '{model_type}'.")

    if hasattr(module, model_class_name):
        return getattr(module, model_class_name)
    else:
        # Fallback: Try retrieving the class from the transformers package
        import transformers
        if hasattr(transformers, model_class_name):
            return getattr(transformers, model_class_name)

    raise ImportError(
        f"The model class '{model_class_name}' was not found for model_type '{model_type}'."
    )
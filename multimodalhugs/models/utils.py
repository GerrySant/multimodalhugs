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

class EncoderWrapper(nn.Module):
    """
    General-purpose encoder wrapper for multimodal models in MultimodalHugs.

    This wrapper calls `model.input_to_encoder_outputs(**kwargs)` and filters
    the incoming keyword arguments to match that method's signature.
    It ensures compatibility with Hugging Face generation logic:
        model.get_encoder() → encoder(**inputs) → encoder_outputs
    """
    def __init__(self, model):
        super().__init__()

        # Check requirements early
        if not hasattr(model, "input_to_encoder_outputs") or not callable(model.input_to_encoder_outputs):
            raise AttributeError(
                f"[MultimodalHugs Error] The model '{model.__class__.__name__}' does not implement "
                f"`input_to_encoder_outputs()`. Encoder-decoder models should either:\n"
                "  1. Define `self.encoder` directly (as in Hugging Face encoder-decoder models), OR\n"
                "  2. Implement `input_to_encoder_outputs(self, **kwargs)` that returns encoder outputs.\n\n"
                "Without one of these, generation and saving cannot work properly."
            )

        # Store a weak reference (breaks circular reference for saving)
        self._model_ref = model
        self._sig = inspect.signature(model.input_to_encoder_outputs)

    def forward(self, *args, **kwargs):
        """
        Delegates the call to model.input_to_encoder_outputs(), filtering keyword
        arguments to those expected by that method.
        """
        model = self._model_ref
        sig = self._sig

        # Filter only valid kwargs
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return model.input_to_encoder_outputs(*args, **valid_kwargs)


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
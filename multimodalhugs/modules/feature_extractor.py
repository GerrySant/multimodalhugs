# Standard libraries
import logging
import importlib
from pathlib import Path

# Third-party libraries
import torch
import torch.nn as nn
from transformers import (
    CLIPConfig, CLIPModel, M2M100Config, M2M100Model, 
    PreTrainedModel, PretrainedConfig
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

# Other
from typing import Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Factory function to create the appropriate feature extractor class based on feature_extractor_type
def get_feature_extractor_class(feature_extractor_type: str):
    """
    Retrieves both the model (feature_extractor) class and the configuration class for the given model type.
    It first attempts to import from the 'transformers.models.{module_name}' module (where module_name is
    feature_extractor_type with dashes replaced by underscores) and falls back to the top-level transformers package if needed.

    Args:
        feature_extractor_type (str): The model type (e.g., 'bert', 't5', etc.).

    Returns:
        tuple: A tuple (model_class, config_class).

    Raises:
        ValueError: If feature_extractor_type is not found in the mapping dictionaries.
        ImportError: If either the model class or the configuration class cannot be imported.
    """
    # Retrieve config class
    if feature_extractor_type not in CONFIG_MAPPING_NAMES:
        raise ValueError(
            f"Unknown model type '{feature_extractor_type}'. Available options: {list(CONFIG_MAPPING_NAMES.keys())}"
        )
    config_class_name = CONFIG_MAPPING_NAMES[feature_extractor_type]
    module_name = feature_extractor_type.replace("-", "_")
    
    config_class = None
    try:
        config_module = importlib.import_module(f"transformers.models.{module_name}")
        if hasattr(config_module, config_class_name):
            config_class = getattr(config_module, config_class_name)
    except ImportError:
        pass

    if config_class is None:
        # Fallback to top-level transformers package
        import transformers
        if hasattr(transformers, config_class_name):
            config_class = getattr(transformers, config_class_name)
        else:
            raise ImportError(
                f"The configuration class '{config_class_name}' was not found for feature_extractor_type '{feature_extractor_type}'."
            )
    
    # Retrieve model class
    if feature_extractor_type not in MODEL_MAPPING_NAMES:
        raise ValueError(
            f"Unknown model type '{feature_extractor_type}'. Available options: {list(MODEL_MAPPING_NAMES.keys())}"
        )
    model_class_name = MODEL_MAPPING_NAMES[feature_extractor_type]
    
    model_class = None
    try:
        model_module = importlib.import_module(f"transformers.models.{module_name}")
        if hasattr(model_module, model_class_name):
            model_class = getattr(model_module, model_class_name)
    except ImportError:
        pass

    if model_class is None:
        # Fallback to top-level transformers package
        import transformers
        if hasattr(transformers, model_class_name):
            model_class = getattr(transformers, model_class_name)
        else:
            raise ImportError(
                f"The model class '{model_class_name}' was not found for feature_extractor_type '{feature_extractor_type}'."
            )
    
    return model_class, config_class

class FeatureExtractor(nn.Module):
    """
    A wrapper for feature extractors.

    Note:
        This class is currently set up for loading pre-trained modules. 
        TODO: Further refinement is needed to:
            - Custom model configuration.
            - Support scratch initialization.
            - Enable automatic general functionality.
    """
    def __init__(self, feature_extractor_type: str, pretrained_module: Optional[Union[str, Path]] = None, config = None):
        super(FeatureExtractor, self).__init__()

        self.feature_extractor_type = feature_extractor_type
        if self.feature_extractor_type:
            FeatureExtractorClass, FeatureExtractorConfigClass = get_feature_extractor_class(self.feature_extractor_type)
            if pretrained_module is not None:
                self.feature_extractor = FeatureExtractorClass.from_pretrained(pretrained_module)
            else:
                self.feature_extractor = FeatureExtractorClass(config)
            
            # Special handling for CLIPModel instances
            if isinstance(self.feature_extractor, CLIPModel):
                self.feature_extractor.text_model = None
                self.feature_extractor.text_projection = None
        else:
            self.feature_extractor = None

    def forward(self, x):
        # Shape of x: [B, T, C, H, W]
        if self.feature_extractor_type == "clip":
            B, T, _, _, _, = x.shape
            x = torch.flatten(x, start_dim=0, end_dim=1) # [B, T, C, H, W] -> [(B x T), C, H, W]
            x = self.feature_extractor.get_image_features(pixel_values=x)
            x = torch.unflatten(x, 0, (B, T)) # [(B x T), E] -> [B, T, E]
        return x

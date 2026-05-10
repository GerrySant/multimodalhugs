# Standard libraries
import logging
import importlib
from pathlib import Path

# Third-party libraries
import torch
import torch.nn as nn
from transformers import (
    CLIPConfig, CLIPVisionConfig, CLIPVisionModelWithProjection,
    M2M100Config, M2M100Model,
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

def load_pretrained_feature_extractor(feature_extractor_type: str, pretrained_module: Union[str, Path]):
    """
    Load a pretrained inner model for the given feature extractor type.

    Called exclusively from ``build_model`` at setup time to obtain pretrained
    weights that are then copied into the assembled model. Never called from
    ``FeatureExtractor.__init__``, keeping ``__init__`` structure-only and
    compatible with transformers 5.x ``init_empty_weights()`` context.
    """
    if feature_extractor_type == "clip":
        return CLIPVisionModelWithProjection.from_pretrained(pretrained_module)
    FeatureExtractorClass, _ = get_feature_extractor_class(feature_extractor_type)
    return FeatureExtractorClass.from_pretrained(pretrained_module)


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
            if self.feature_extractor_type == "clip":
                # Use CLIPVisionModelWithProjection instead of CLIPModel.
                # CLIPModel nullification (text_model=None, text_projection=None) crashes
                # in transformers 5.x because initialize_weights() unconditionally accesses
                # text_projection.weight via _init_weights.  CLIPVisionModelWithProjection
                # has only the vision encoder and visual_projection — no text branch at all.
                #
                # Weight loading via from_pretrained() is NOT done here. __init__ is
                # responsible for model structure only. build_model() loads pretrained weights
                # and copies them explicitly; from_pretrained(saved_path) loads them from the
                # combined checkpoint. This makes __init__ compatible with transformers 5.x
                # init_empty_weights() used during from_pretrained loading.
                if isinstance(config, CLIPConfig):
                    vision_config = config.vision_config
                elif isinstance(config, dict):
                    vision_config = CLIPVisionConfig(**config)
                elif config is None:
                    vision_config = CLIPVisionConfig()
                else:
                    vision_config = config  # already a CLIPVisionConfig
                self.feature_extractor = CLIPVisionModelWithProjection(vision_config)
            else:
                FeatureExtractorClass, ConfigClass = get_feature_extractor_class(self.feature_extractor_type)
                if isinstance(config, dict):
                    config = ConfigClass.from_dict(config)
                elif config is None:
                    config = ConfigClass()
                self.feature_extractor = FeatureExtractorClass(config)
            # Propagate FSDP parallelism metadata from the inner model so that
            # MultiModalEmbedderModel._init_feature_extractor can read them transparently.
            self._no_split_modules = list(getattr(self.feature_extractor, "_no_split_modules", None) or [])
            self._keep_in_fp32_modules = list(getattr(self.feature_extractor, "_keep_in_fp32_modules", None) or [])
        else:
            self.feature_extractor = None

    def forward(self, x):
        # Shape of x: [B, T, C, H, W]
        if self.feature_extractor_type == "clip":
            B, T, _, _, _ = x.shape
            x = torch.flatten(x, start_dim=0, end_dim=1)  # [B, T, C, H, W] -> [(B x T), C, H, W]
            x = self.feature_extractor(pixel_values=x).image_embeds  # [(B x T), E]
            x = torch.unflatten(x, 0, (B, T))  # [(B x T), E] -> [B, T, E]
        return x

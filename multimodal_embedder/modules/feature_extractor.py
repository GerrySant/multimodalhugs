import torch
import torch.nn as nn
import logging

from typing import Union, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import (
    CLIPConfig, CLIPModel, M2M100Config, M2M100Model, 
    PreTrainedModel, PretrainedConfig
)

# Factory function to create the appropriate feature extractor class based on feature_extractor_type
def get_feature_extractor_class(feature_extractor_type):
    if feature_extractor_type == "clip": # The actual version only supports CLIP as feature extractor
        return CLIPModel, CLIPConfig
    else:
        raise ValueError(f"Unknown feature extractor type: {feature_extractor_type}")

class FeatureExtractor(nn.Module):
    def __init__(self, feature_extractor_type: str, pretrained_module: Optional[Union[str, Path]] = None, config = None):
        super(FeatureExtractor, self).__init__()

        self.feature_extractor_type = feature_extractor_type
        if self.feature_extractor_type:
            FeatureExtractorClass, FeatureExtractorConfigClass = get_feature_extractor_class(self.feature_extractor_type)
            if pretrained_module is not None:
                self.feature_extractor = FeatureExtractorClass.from_pretrained(pretrained_module)
            else:
                self.feature_extractor = FeatureExtractorClass(config)
                
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

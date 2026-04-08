import pytest

from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig


# max_length and use_backbone_max_length were removed from MultiModalEmbedderConfig in
# the transformers 5.x compatibility update (see docs/transformers_compatibility.md §8).
# Generation length is now managed via model.generation_config.max_length, which is the
# standard HuggingFace 5.x pattern. The previous config-level tests were removed along
# with the parameters themselves.

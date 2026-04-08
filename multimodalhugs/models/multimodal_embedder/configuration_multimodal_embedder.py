# Standard Library Imports
import logging
import math
import importlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union

# Third-Party Imports
import torch
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers import (
    M2M100ForConditionalGeneration,
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from accelerate.utils import find_tied_parameters
from ruamel.yaml import YAML

# Local Application Imports
from multimodalhugs.models.utils import get_backbone_config_class, get_backbone_model_class
from multimodalhugs.utils.registry import register_model
from multimodalhugs.modules import MultimodalMapper, FeatureExtractor, get_feature_extractor_class
from multimodalhugs.modules.utils import set_module_parameters, extend_all_embeddings_and_lm_head, merge_modalities, merge_modalities_mask_correction
from multimodalhugs.utils import serialize_config

logger = logging.getLogger(__name__)


class MultiModalEmbedderConfig(PretrainedConfig):
    """
    This class extends transformers.PretrainedConfig to configure the MultiModalEmbedderModel model class.

    This configuration includes parameters for the feature extractor, visual-language mapping, and backbone model.

    Refer to the [transformers.PretrainedConfig documentation](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/configuration#transformers.PretrainedConfig) to specify arguments of the parent class.

    Parameters
    ----------
    model_type : str
        Name of the model type.
    d_model : int, optional
        Dimension of the model.
    feat_dim : int
        Feature extractor output dimension. If offline features are used, the
        dimensionality of the extracted features.
    feature_extractor_type : str, optional
        Type of feature extractor to use.
    feature_extractor_config : dict, optional
        Hyperparameters for the feature extractor. Parameters not specified
        default to those of the feature extractor class. If loading from a
        pretrained model, these values are overridden unless explicitly set.
    pretrained_feature_extractor : str, optional
        Path or identifier for a pretrained feature extractor.
    freeze_feature_extractor : bool
        Whether to freeze feature extractor parameters during training.
    multimodal_mapper_type : str, optional
        Type of multimodal mapper: one of {"linear", "adapter", "cnn_adapter"}.
    multimodal_mapper_layer_norm_before : bool
        Whether to apply LayerNorm before the mapper.
    multimodal_mapper_layer_norm : bool
        Whether to apply LayerNorm inside the mapper.
    multimodal_mapper_activation : bool
        Whether to apply ReLU at the mapper output.
    multimodal_mapper_factor : int, optional
        Overparameterization factor used by the adapter-based mapper.
    multimodal_mapper_dropout : float, optional
        Dropout probability for the multimodal mapper.
    adapter_ksize : tuple of int, optional
        Kernel size(s) for the cnn_adapter mapper.
    adapter_stride : tuple of int, optional
        Stride(s) for the cnn_adapter mapper.
    freeze_multimodal_mapper : bool
        Whether to freeze multimodal mapper parameters during training.
    backbone_used_vocab_size : int, optional
        Original backbone vocab size (excluding pruned/garbage embeddings).
    backbone_type : str
        Model type to use as the backbone.
    backbone_config : dict, optional
        Hyperparameters for the backbone model, following the same logic as
        feature_extractor_config.
    pretrained_backbone : str, optional
        Identifier or path of the pretrained backbone.
    backbone_tied_weights_keys : Any, optional
        Parameter keys whose weights are tied within the backbone.
    freeze_backbone : bool
        Whether to freeze the entire backbone during training.
    freeze_encoder_embed_tokens : bool
        Whether to freeze encoder embedding parameters.
    freeze_decoder_embed_tokens : bool
        Whether to freeze decoder embedding parameters.
    freeze_lm_head : bool
        Whether to freeze the LM head.
    is_encoder_decoder : bool
        Whether the model follows the encoder-decoder architecture.
    decoder_start_token_id : int, optional
        Start token for decoding (if different from BOS).
    pad_token_id : int, optional
        Padding token ID.
    bos_token_id : int, optional
        Beginning-of-stream token ID.
    eos_token_id : int, optional
        End-of-stream token ID.
    ```python
    >>> from multimodalhugs.models.multimodal_embedder.configuration_multimodal_embedder import MultiModalEmbedderConfig
    >>> config = MultiModalEmbedderConfig(d_model=1024, backbone_type="m2m_100")
    >>> print(config.backbone_type)
    m2m_100
    ```"""

    model_type = "multimodal_embedder"

    def __init__(
        self,
        model_type: str = "multimodal_embedder",
        d_model: Optional[int] = None,
        feat_dim: int = 512,
        feature_extractor_type: Optional[str] = None,
        feature_extractor_config: Optional[Dict[str, Any]] = None,
        pretrained_feature_extractor: Optional[str] = None,
        freeze_feature_extractor: bool = False,
        multimodal_mapper_type: Optional[str] = None,
        multimodal_mapper_layer_norm_before: bool = False,
        multimodal_mapper_layer_norm: bool = False,
        multimodal_mapper_activation: bool = False,
        multimodal_mapper_factor: Optional[int] = None,
        multimodal_mapper_dropout: Optional[float] = None,
        adapter_ksize: Optional[Tuple[int, ...]] = None,
        adapter_stride: Optional[Tuple[int, ...]] = None,
        freeze_multimodal_mapper: bool = False,
        backbone_used_vocab_size: Optional[int] = None,
        backbone_type: str = "m2m_100",
        backbone_config: Optional[Dict[str, Any]] = None,
        pretrained_backbone: Optional[str] = None,
        backbone_tied_weights_keys: Optional[Any] = None,
        freeze_backbone: bool = False,
        freeze_encoder_embed_tokens: bool = False,
        freeze_decoder_embed_tokens: bool = False,
        freeze_lm_head: bool = False,
        is_encoder_decoder: bool = True,
        decoder_start_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs):

        # Pass remaining arguments to the parent (e.g. name, hashes, revisions)
        super().__init__(**kwargs)

        # Assign all fields to self
        self.model_type = model_type
        self.d_model = d_model
        self.feat_dim = feat_dim
        self.feature_extractor_type = feature_extractor_type
        self.feature_extractor_config = feature_extractor_config
        self.pretrained_feature_extractor = pretrained_feature_extractor
        self.freeze_feature_extractor = freeze_feature_extractor

        self.multimodal_mapper_type = multimodal_mapper_type
        self.multimodal_mapper_layer_norm_before = multimodal_mapper_layer_norm_before
        self.multimodal_mapper_layer_norm = multimodal_mapper_layer_norm
        self.multimodal_mapper_activation = multimodal_mapper_activation
        self.multimodal_mapper_factor = multimodal_mapper_factor
        self.multimodal_mapper_dropout = multimodal_mapper_dropout
        self.adapter_ksize = adapter_ksize
        self.adapter_stride = adapter_stride
        self.freeze_multimodal_mapper = freeze_multimodal_mapper

        self.backbone_used_vocab_size = backbone_used_vocab_size
        self.backbone_type = backbone_type
        self.backbone_config = backbone_config
        self.pretrained_backbone = pretrained_backbone
        self.backbone_tied_weights_keys = backbone_tied_weights_keys
        self.freeze_backbone = freeze_backbone
        self.freeze_encoder_embed_tokens = freeze_encoder_embed_tokens
        self.freeze_decoder_embed_tokens = freeze_decoder_embed_tokens
        self.freeze_lm_head = freeze_lm_head

        self.is_encoder_decoder = is_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # additional changes

        self.is_encoder_decoder = True

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

        if self.backbone_config is not None:
            backbone_config_class = get_backbone_config_class(self.backbone_type)
            self.backbone_config = backbone_config_class(**self.backbone_config)

        elif self.pretrained_backbone is not None:
            self.backbone_config = AutoConfig.from_pretrained(self.pretrained_backbone)

        if self.backbone_config is not None:
            # tie_encoder_decoder was removed from all seq2seq model configs in
            # transformers 5.x and is no longer a supported attribute.
            self.tie_word_embeddings = getattr(self.backbone_config, "tie_word_embeddings", False)

        if self.feature_extractor_type is not None:
            feature_xtractor_config_class = get_feature_extractor_class(self.feature_extractor_type)[1]
            self.feature_extractor_config = feature_xtractor_config_class(**self.feature_extractor_config if
                                                                          self.feature_extractor_config is not None else {})
        else:
            self.feature_extractor_config = None

        self.adapter_ksize = eval(self.adapter_ksize) if isinstance(self.adapter_ksize, str) else self.adapter_ksize
        self.adapter_stride = eval(self.adapter_stride) if isinstance(self.adapter_stride, str) else self.adapter_stride
        self.bos_token_id = self.bos_token_id if self.bos_token_id is not None else self.decoder_start_token_id

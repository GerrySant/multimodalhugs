# Standard Library Imports
import logging
import math
import importlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union

# Third-Party Imports
import torch
from transformers.models.auto.modeling_auto import MODEL_WITH_LM_HEAD_MAPPING_NAMES
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
from multimodalhugs.models import EncoderWrapper, get_backbone_config_class, get_backbone_model_class
from multimodalhugs.utils.registry import register_model
from multimodalhugs.modules import MultimodalMapper, FeatureExtractor, get_feature_extractor_class
from multimodalhugs.modules.utils import set_module_parameters, extend_all_embeddings_and_lm_head, merge_modalities, merge_modalities_mask_correction
from multimodalhugs.utils import serialize_config

logger = logging.getLogger(__name__)

@dataclass
class MultiModalEmbedderConfig(PretrainedConfig):
    r"""
    This class extends transformers.PretrainedConfig to configure the MultiModalEmbedderModel model class.

    This configuration includes parameters for the feature extractor, visual-language mapping, and backbone model.

    Refer to the [transformers.PretrainedConfig documentation](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/configuration#transformers.PretrainedConfig) to specify arguments of the parent class.
    """

    model_type: str = field(
        default="multimodal_embedder", metadata={"help": "Name of the model to be used."}
    )
    d_model: Optional[int] = field(
        default=None, metadata={"help": "Dimention of the model"}
    )
    feat_dim: int = field(
        default=512, metadata={"help": "Dimention of the Feature Extractor output. If features are extracted off-line, the dimentionality of features."}
    )
    feature_extractor_type: Optional[str] = field(
        default=None, metadata={"help": "Feature Extractor type to be used."}
    )
    feature_extractor_config: Optional[Dict[str, Any]] = field(
        default=None, metadata={
            "help": "Hyperparameters of the model class specified in feature_extractor_type. Those not specified are assumed to be the default values of the model class.", 
            "extra_info": "In case of initializing the feature_extractor from a pre-trained model, the feature_extractor parameters will be defined automatically, modifying only those specified under this field."
            },
    )
    pretrained_feature_extractor: Optional[str] = field(
        default=None, metadata={"help": "Pretrained Feature Extractor or path to the Pretrained Feature Extractor checkpoint."}
    )
    freeze_feature_extractor: bool = field(
        default=False, metadata={"help": "if True, the feature_extractor parameters are frozen during training."}
    )
    multimodal_mapper_type: Optional[str] = field(
        default=None, metadata={"help": "Chose the Multimodal Mapper type. Options: 'linear', 'adapter', 'cnn_adapter'"}
    )
    multimodal_mapper_layer_norm_before: bool = field(
        default=False, metadata={"help": "if True, adds a LayerNorm before the multimodal_mapper"}
    )
    multimodal_mapper_layer_norm: bool = field(
        default=False, metadata={"help": "if True, adds a LayerNorm inside the multimodal_mapper"}
    )
    multimodal_mapper_activation: bool = field(
        default=False, metadata={"help": "if True, applies a ReLu at the multimodal_mapper output"}
    )
    multimodal_mapper_factor: Optional[int] = field(
        default=None,
        metadata={"help": "If specified, use an adapter as Multimodal mapper whose overparameterization is given by the given factor"}
    )
    multimodal_mapper_dropout: Optional[float] = field(
        default=None, metadata={"help": "Dropout probabilty for the multimodal_mapper"}
    )
    adapter_ksize: Optional[Tuple[int, ...]] = field(
        default=None,
        metadata={"help": "If specified, indicates kernel sizes to use on the cnn_adapter. Can be a single integer or a tuple of integers."}
    )
    adapter_stride: Optional[Tuple[int, ...]] = field(
        default=None,
        metadata={"help": "If specified, indicates the stride to use by the cnn_adapter. Can be a single integer or a tuple of integers."}
    )
    freeze_multimodal_mapper: bool = field(
        default=False, metadata={"help": "if True, the multimodal_mapper parameters are frozen during training."}
    )
    backbone_used_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "Original vocab_size of the backbone excluding garbage embeddings"}
    )
    backbone_type: str = field(
        default="m2m_100", metadata={"help": "Type of the model to be used as a backbone"}
    )
    backbone_config: Optional[Dict[str, Any]] = field(
        default=None, metadata={
            "help": "Hyperparameters of the model class specified in backbone_type. Those not specified are assumed to be the default values of the model class.", 
            "extra_info": "In case of initializing the backbone from a pre-trained model, the backbone parameters will be defined automatically, modifying only those specified under this field."
            },
    )
    pretrained_backbone: Optional[str] = field(
        default=None, metadata={"help": "Pretrained Backbone or path to the Pretrained Backbone checkpoint."}
    )
    backbone_tied_weights_keys: Optional[Any] = field(
        default=None, metadata={"help": "Keys of the model parameters that are tied to each other."}
    )
    freeze_backbone: bool = field(
        default=False, metadata={"help": "if True, the backbone parameters are frozen during training."}
    )
    freeze_encoder_embed_tokens: bool = field(
        default=False, metadata={"help": "if True, the encoder.embed_tokens parameters are frozen during training."}
    )
    freeze_decoder_embed_tokens: bool = field(
        default=False, metadata={"help": "if True, the decoder.embed_tokens parameters are frozen during training."}
    )
    freeze_lm_head: bool = field(
        default=False, metadata={"help": "if True, the lm_head parameters are frozen during training."}
    )
    is_encoder_decoder: bool = field(
        default=True, metadata={"help": "Whether the model is used as an encoder/decoder or not."}
    )
    decoder_start_token_id: Optional[int] = field(
        default=None, metadata={"help": "If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token."}
    )
    pad_token_id: Optional[int] = field(
        default=None, metadata={"help": "The id of the _padding_ token."}
    )
    bos_token_id: Optional[int] = field(
        default=None, metadata={"help": "The id of the _beginning-of-stream_ token."}
    )
    eos_token_id: Optional[int] = field(
        default=None, metadata={"help": "The id of the _end-of-stream_ token."}
    )
    max_length: int = field(
        default=1024, metadata={"help": "The maximum target length to use when predicting with the generate method."}
    )

    def __init__(self, **kwargs):
        """
        **Initialize the MultiModalEmbedderConfig.**

        **Args:**
        - `kwargs`: Additional keyword arguments to configure the model.

        **Example:**
        ```python
        config = MultiModalEmbedderConfig(d_model=1024, backbone_type="m2m_100")
        print(config.backbone_type)  # Output: "m2m_100"
        ```
        """
        super().__init__(**kwargs)
        self.is_encoder_decoder = True
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
                
        if kwargs and 'backbone_type' in kwargs and 'backbone_config' in kwargs:
            backbone_type = kwargs['backbone_type']
            BackboneConfigClass = get_backbone_config_class(backbone_type)
            self.backbone_config = BackboneConfigClass(**kwargs.get('backbone_config', {}))

        elif kwargs and 'pretrained_backbone' in kwargs:
            self.backbone_config = AutoConfig.from_pretrained(kwargs['pretrained_backbone'])

        else:
            self.backbone_config = None

        if self.backbone_config is not None:
            self.tie_encoder_decoder = self.backbone_config.tie_encoder_decoder
            self.tie_word_embeddings = self.backbone_config.tie_word_embeddings

        if kwargs and 'feature_extractor_type' in kwargs:
            feature_extractor_type = kwargs['feature_extractor_type']
            FeatureExtractorConfigClass = get_feature_extractor_class(feature_extractor_type)[1]
            self.feature_extractor_config = FeatureExtractorConfigClass(**kwargs.get('feature_extractor_config', {}))
        else:
            self.feature_extractor_config = None

        self.adapter_ksize = eval(self.adapter_ksize) if isinstance(self.adapter_ksize, str) else self.adapter_ksize
        self.adapter_stride = eval(self.adapter_stride) if isinstance(self.adapter_stride, str) else self.adapter_stride
        self.bos_token_id = self.bos_token_id if self.bos_token_id is not None else self.decoder_start_token_id
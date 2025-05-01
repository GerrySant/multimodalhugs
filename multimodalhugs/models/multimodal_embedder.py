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

# Define the custom model class
@register_model("multimodal_embedder")
class MultiModalEmbedderModel(PreTrainedModel):
    """
    **MultiModalEmbedderModel: A Transformer-based multimodal model.**

    This model extends `transformers.PreTrainedModel`, integrating visual and textual 
    inputs using a feature extractor, a Multimodal Mapper (Multimodal Mapper), and 
    a backbone Transformer model.
    """
    config_class = MultiModalEmbedderConfig
    base_model_prefix = "multimodal_embedder"
    is_parallelizable = True
    _keep_in_fp32_modules = []
    _no_split_modules = []

    def __init__(self, config):
        """
        **Initialize the MultiModalEmbedderModel.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        super().__init__(config)
        self._init_feature_extractor(config)
        self._init_multimodal_mapper(config)
        self.decoder_start_token_id = config.decoder_start_token_id
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id if config.bos_token_id is not None else config.decoder_start_token_id
        self._init_backbone(config)
        self.max_length = config.max_length
        self.post_init()

    def _init_feature_extractor(self, config):
        """
        **Initialize the feature extractor.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        if config.feature_extractor_type:
            self.feature_extractor = FeatureExtractor(
                feature_extractor_type=config.feature_extractor_type,
                pretrained_module=config.pretrained_feature_extractor,
                config=config.feature_extractor_config,
            )
            set_module_parameters(self.feature_extractor, freeze=config.freeze_feature_extractor)
        else:
            self.feature_extractor = None

        if self.feature_extractor is not None:
            self.is_parallelizable = self.is_parallelizable and getattr(self.feature_extractor, "is_parallelizable", True)
            self._no_split_modules = self._no_split_modules + (getattr(self.feature_extractor, "_no_split_modules", []) or [])
            self._keep_in_fp32_modules = self._keep_in_fp32_modules + (getattr(self.feature_extractor, "_keep_in_fp32_modules", []) or [])

    def _init_multimodal_mapper(self, config):
        """
        **Initialize the Visual-Language (VL) Mapper.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        if config.multimodal_mapper_type is not None:
            self.multimodal_mapper = MultimodalMapper(
                feat_dim=config.feat_dim,
                output_dim=config.d_model,
                mapping_layer_type=config.multimodal_mapper_type,
                layer_norm_before=config.multimodal_mapper_layer_norm_before,
                adapter_factor=config.multimodal_mapper_factor,
                p_dropout=config.multimodal_mapper_dropout,
                layer_norm=config.multimodal_mapper_layer_norm,
                activation=config.multimodal_mapper_activation,
                adapter_ksize=config.adapter_ksize,
                adapter_stride=config.adapter_stride
            )
            set_module_parameters(self.multimodal_mapper, freeze=config.freeze_multimodal_mapper)
        else:
            self.multimodal_mapper = None

    def _init_backbone(self, config):
        """
        **Initialize the Transformer backbone model.**

        **Args:**
        - `config` (MultiModalEmbedderConfig): Model configuration.
        """
        BackboneModelClass = get_backbone_model_class(config.backbone_type)
        if config.backbone_config is not None:
            self.backbone = BackboneModelClass(config.backbone_config)
        else:
            self.backbone = BackboneModelClass.from_pretrained(config.pretrained_backbone)
        
        if isinstance(config.backbone_tied_weights_keys, list):
            self.backbone._tied_weights_keys = config.backbone_tied_weights_keys

        set_module_parameters(self.backbone, freeze=config.freeze_backbone)
        set_module_parameters(self.get_backbone_encoder.embed_tokens, freeze=config.freeze_encoder_embed_tokens)
        set_module_parameters(self.get_backbone_decoder.embed_tokens, freeze=config.freeze_decoder_embed_tokens)
        set_module_parameters(self.lm_head, freeze=config.freeze_lm_head)
        
        freeze_shared = (
            config.freeze_decoder_embed_tokens or 
            config.freeze_encoder_embed_tokens or 
            config.freeze_lm_head
        )
        set_module_parameters(self.get_shared, freeze=freeze_shared, verbose=False)
        self.is_parallelizable = self.is_parallelizable and getattr(self.backbone, "is_parallelizable", True)
        self._no_split_modules = self._no_split_modules + (getattr(self.backbone, "_no_split_modules", []) or [])
        self._keep_in_fp32_modules = self._keep_in_fp32_modules + (getattr(self.backbone, "_keep_in_fp32_modules", []) or [])

    def get_input_embeddings(self):
        """
        **Retrieve the input embeddings.**

        **Returns:**
        - `torch.nn.Module`: Input embedding layer.
        """
        if hasattr(self.backbone, 'shared'):
            return self.backbone.shared
        elif hasattr(self.backbone, 'model'):
            if hasattr(self.backbone.model, 'shared'):
                return getattr(self.backbone.model, 'shared', None) 
        else:
            return None

    def set_input_embeddings(self, value):
        """
        **Set new input embeddings.**

        **Args:**
        - `value` (torch.nn.Module): New embedding module.
        """
        # Set 'shared' attribute
        if hasattr(self.backbone, 'shared'):
            self.backbone.shared = value
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'shared'):
            self.backbone.model.shared = value
        else:
            raise AttributeError("Neither 'shared' nor 'model.shared' exists in the backbone.")
        
        # Set 'encoder.embed_tokens'
        encoder = (
            getattr(self.backbone, 'encoder', None) or
            getattr(self.backbone.model, 'encoder', None)
        )
        if encoder and hasattr(encoder, 'embed_tokens'):
            encoder.embed_tokens = value

        # Set 'decoder.embed_tokens'
        decoder = (
            getattr(self.backbone, 'decoder', None) or
            getattr(self.backbone.model, 'decoder', None)
        )
        if decoder and hasattr(decoder, 'embed_tokens'):
            decoder.embed_tokens = value

    def get_output_embeddings(self):
        """
        **Retrieve the output embedding layer (LM Head).**

        **Returns:**
        - `torch.nn.Module`: LM head layer.
        """
        return self.lm_head

    @property
    def lm_head(self):
        """
        **Retrieve the language modeling head.**

        **Returns:**
        - `torch.nn.Module`: LM head module.
        """
        if hasattr(self.backbone, 'lm_head'):
            return self.backbone.lm_head
        return None

    @property
    def get_backbone_encoder(self):
        """
        **Retrieve the encoder module from the backbone.**

        This method checks if the backbone model has a direct `encoder` attribute.
        If not, it assumes the backbone has a `.model` submodule containing the encoder.

        **Returns:**
        - `torch.nn.Module`: The encoder module of the backbone model.
        """
        return self.backbone.encoder if hasattr(self.backbone, 'encoder') else self.backbone.model.encoder

    @property
    def get_backbone_decoder(self):
        """
        **Retrieve the decoder module from the backbone.**

        Similar to `get_backbone_encoder`, this method checks if the backbone model has a 
        direct `decoder` attribute. If not, it assumes the backbone has a `.model` submodule 
        containing the decoder.

        **Returns:**
        - `torch.nn.Module`: The decoder module of the backbone model.
        """
        return self.backbone.decoder if hasattr(self.backbone, 'decoder') else self.backbone.model.decoder
    
    @property
    def get_shared(self):
        """
        **Retrieve the shared embedding layer of the backbone.**

        This method returns the shared embedding layer, if present, which is commonly used 
        to tie input and output embeddings in Transformer-based architectures.

        **Returns:**
        - `torch.nn.Module` or `None`: The shared embedding layer, if available.
        """
        if hasattr(self.backbone, 'shared'):
            return self.backbone.shared
        elif hasattr(self.backbone, 'model'):
            if hasattr(self.backbone.model, 'shared'):
                return getattr(self.backbone.model, 'shared', None) 
        else:
            return None

    @classmethod
    def build_model(cls, **kwargs):
        """
        Build the model instance using provided configuration parameters.
        
        Expected common keys:
          - src_tokenizer: the source tokenizer.
          - tgt_tokenizer: the target tokenizer.
          - config_path: the path to the YAML configuration file.
          - new_vocab_tokens: a list of new vocabulary tokens.
        
        Plus any extra keys defined in the config.
        """
        # Extract common arguments if needed:
        src_tokenizer = kwargs.pop("src_tokenizer")
        tgt_tokenizer = kwargs.pop("tgt_tokenizer")
        config_path = kwargs.pop("config_path", None)
        new_vocab_tokens = kwargs.pop("new_vocab_tokens", [])

        if src_tokenizer is None or tgt_tokenizer is None:
            raise ValueError("Please provide the src_tokenizer and the tgt_tokenizer in case the dataset used does not have these as a parameter.")
            
        cfg = kwargs  # or merge with defaults, etc.
        if not isinstance(cfg, PretrainedConfig):
            cfg = cls.config_class.from_dict(serialize_config(cfg))
        else:
            cfg = cfg

        # Load backbone model & config
        BackboneModelClass = get_backbone_model_class(cfg.backbone_type)
        if cfg.pretrained_backbone is not None:
            backbone = BackboneModelClass.from_pretrained(cfg.pretrained_backbone)
            cfg.backbone_config = AutoConfig.from_pretrained(cfg.pretrained_backbone)
        else:
            backbone = BackboneModelClass(cfg.backbone_config)
        
        cfg.d_model = cfg.d_model or cfg.backbone_config.d_model
        cfg.decoder_start_token_id = cfg.decoder_start_token_id or cfg.backbone_config.decoder_start_token_id
        cfg.backbone_tied_weights_keys = cfg.backbone_tied_weights_keys or find_tied_parameters(backbone)[0]
        backbone._tied_weights_keys = cfg.backbone_tied_weights_keys

        # Determine EOS and PAD token indices
        pad_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.pad_token)
        bos_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.bos_token)
        eos_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.eos_token)

        # Update configuration with these values if not already defined
        cfg.pad_token_id = cfg.pad_token_id or pad_token_id
        cfg.bos_token_id = cfg.bos_token_id or bos_token_id
        cfg.eos_token_id = cfg.eos_token_id or eos_token_id
        tokenizer_vocab_size = getattr(src_tokenizer, "total_vocab_size", src_tokenizer.vocab_size)
        #cfg.backbone_used_vocab_size = cfg.backbone_used_vocab_size or (tokenizer_vocab_size - len(new_vocab_tokens))

        # Update YAML configuration file
        if config_path:
            yaml = YAML()
            with open(config_path, 'r') as file:
                config_data = yaml.load(file)

            # Add or update these keys in the YAML file
            config_data['model']['d_model'] = cfg.d_model
            config_data['model']['pad_token_id'] = cfg.pad_token_id
            config_data['model']['bos_token_id'] = cfg.bos_token_id
            config_data['model']['eos_token_id'] = cfg.eos_token_id
            config_data['model']['decoder_start_token_id'] = cfg.decoder_start_token_id
            
            # Save the updated configuration back to the file
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file)

        backbone, new_vocab_size = extend_all_embeddings_and_lm_head(backbone=backbone, num_new_tokens=len(new_vocab_tokens), verbose=True)
        cfg.backbone_config.vocab_size = new_vocab_size

        # Create an instance of the model
        model = cls(config=cfg)

        # Copy the weights from the backbone instance to the model.backbone
        model.backbone.load_state_dict(backbone.state_dict())

        # Converts all tensors in the model to contiguous
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        return model

    def forward(
        self,
        input_frames: Optional[torch.LongTensor] = None,
        encoder_prompt: Optional[torch.LongTensor] = None,
        encoder_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """
        **Forward pass of the MultiModalEmbedderModel.**

        This method performs the forward propagation of the model, processing multimodal 
        inputs including textual and video-based features. The method integrates visual 
        embeddings, applies the Multimodal Mapper (Multimodal Mapper), and processes text 
        tokens through the Transformer backbone.

        ### **Args:**
        - `input_frames` (Optional[torch.LongTensor], shape: `(B, N_frames, C, W, H)`):  
        The batch of video input frames, where:
            - `B` = batch size  
            - `N_frames` = number of frames per sample  
            - `C` = number of channels  
            - `W` = frame width  
            - `H` = frame height  

        - `encoder_prompt` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
        A prompt consisting of tokenized text that is prepended to the model's input.

        - `encoder_prompt_length_padding_mask` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
        Mask to indicate padding tokens in the encoder prompt.

        - `input_ids` (Optional[torch.LongTensor], shape: `(B, S_text)`):  
        Tokenized input sequence, where:
            - `S_text` = sequence length (number of tokens)  
        Padding tokens will be ignored.

        - `attention_mask` (Optional[torch.Tensor], shape: `(B, N_frames)`):  
        A mask that indicates which tokens or frames should be attended to (`1`) and 
        which should be ignored (`0`).

        - `decoder_input_ids` (Optional[torch.LongTensor], shape: `(B, T_text)`):  
        Input IDs for the decoder during training or inference.  
        - If using teacher forcing, should have the format: `['<prompt_1>', '<prompt_2>,  ..., <prompt_N>', '<token_a>', '<token_b>', '<token_c>']`.  
        - In generation mode: `['<prompt_1>', '<prompt_2>,  ..., <prompt_N>']`.

        - `decoder_attention_mask` (Optional[torch.LongTensor], shape: `(B, T_text)`):  
        Mask for decoder inputs, where `0` indicates padding elements.

        - `head_mask` (Optional[torch.Tensor], shape: `(num_layers, num_heads)`):  
        Mask for attention heads in the encoder.

        - `decoder_head_mask` (Optional[torch.Tensor], shape: `(num_layers, num_heads)`):  
        Mask for attention heads in the decoder.

        - `cross_attn_head_mask` (Optional[torch.Tensor], shape: `(num_layers, num_heads)`):  
        Mask for cross-attention heads in the decoder.

        - `encoder_outputs` (Optional[Tuple[Tuple[torch.FloatTensor]]]):  
        Precomputed encoder outputs, useful when using cached values for efficiency.

        - `past_key_values` (Optional[Tuple[Tuple[torch.FloatTensor]]]):  
        Cached past key-value pairs for decoder self-attention and cross-attention.  
        Used to speed up autoregressive generation.

        - `inputs_embeds` (Optional[torch.FloatTensor], shape: `(B, S_text, hidden_dim)`):  
        Precomputed input embeddings instead of `input_ids`.

        - `decoder_inputs_embeds` (Optional[torch.FloatTensor], shape: `(B, T_text, hidden_dim)`):  
        Precomputed embeddings for decoder inputs.

        - `labels` (Optional[torch.LongTensor], shape: `(B, T_text)`):  
        Target text token IDs, required during training.  
        Should follow the format: `['<prompt_1>', '<prompt_2>,  ..., <prompt_N>', '<token_a>', '<token_b>', '<token_c>', '</s>']`.

        - `use_cache` (Optional[bool], default=`None`):  
        If `True`, enables the use of `past_key_values` for faster decoding.

        - `output_attentions` (Optional[bool], default=`None`):  
        If `True`, the model outputs attention scores.

        - `output_hidden_states` (Optional[bool], default=`None`):  
        If `True`, the model outputs hidden states.

        - `return_dict` (Optional[bool], default=`None`):  
        If `True`, returns a `Seq2SeqLMOutput` instead of a tuple.

        ### **Returns:**
        - `Union[Tuple[torch.Tensor], Seq2SeqLMOutput]`:  
        The model output, which includes:
            - `logits` (torch.Tensor, shape `(B, T_text, vocab_size)`) → Model's output token probabilities.
            - `past_key_values` (Optional[Tuple[Tuple[torch.FloatTensor]]]) → Cached attention states (if `use_cache=True`).
            - `decoder_hidden_states` (Optional[Tuple[torch.FloatTensor]]) → Hidden states of the decoder (if `output_hidden_states=True`).
            - `decoder_attentions` (Optional[Tuple[torch.FloatTensor]]) → Attention scores of the decoder (if `output_attentions=True`).

        ### **Processing Steps:**
        1. **Input Embedding:**  
        - If `inputs_embeds` is not provided, compute it using `feature_extractor(input_frames)`.
        - If a Multimodal Mapper (`multimodal_mapper`) is present, apply it to the embeddings.
        
        2. **Modality Merging:**  
        - Combine `inputs_embeds` with the `encoder_prompt`, if provided.
        - Use the `merge_modalities()` function to ensure proper alignment.
        
        3. **Transformer Backbone Processing:**  
        - The processed embeddings are fed into the backbone Transformer model.
        
        4. **Output Generation:**  
        - The model produces token probabilities (`logits`) and optionally outputs attention states.

        ### **Example Usage:**
        ```python
        model = MultiModalEmbedderModel(config)
        input_frames = torch.randn(4, 16, 3, 224, 224)  # Batch of 4 video clips
        input_ids = torch.randint(0, 50265, (4, 20))  # Random token IDs
        labels = torch.randint(0, 50265, (4, 20))

        outputs = model.forward(input_frames=input_frames, input_ids=input_ids, labels=labels)
        print(outputs.logits.shape)  # Output: (4, 20, 50265)
        ```
        """

        if encoder_outputs is None:
            if labels is not None: 
                # During training, use backbone method to create decoder_input_ids from labels
                decoder_input_ids = None
                decoder_attention_mask = None

            if inputs_embeds is None and input_frames is not None:
                if self.feature_extractor is None:
                    inputs_embeds = input_frames
                else:
                    inputs_embeds = self.feature_extractor(input_frames)

            if self.multimodal_mapper is not None and inputs_embeds is not None:
                inputs_embeds, attention_mask = self.multimodal_mapper(inputs_embeds, attention_mask)

            if inputs_embeds is None:
                inputs_embeds = self.get_backbone_encoder.embed_tokens(input_ids)
                input_ids = None

            inputs_embeds, attention_mask = merge_modalities(
                x=inputs_embeds, 
                padding_mask=attention_mask, 
                prompt=encoder_prompt, 
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens, 
                pad_idx=self.pad_token_id, 
                eos_idx=self.eos_token_id, 
            )
        else:
            if self.multimodal_mapper is not None:
                attention_mask = self.multimodal_mapper.mask_correction(attention_mask)

            # When encoder_outputs is not None, we still have to correct the mask with the proper
            attention_mask = merge_modalities_mask_correction(
                padding_mask=attention_mask, 
                prompt=encoder_prompt, 
                prompt_length_padding_mask=encoder_prompt_length_padding_mask,
                embeddings_module=self.get_backbone_encoder.embed_tokens, 
                pad_idx=self.pad_token_id, 
                eos_idx=self.eos_token_id, 
            )
            
        outputs = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            head_mask = head_mask,
            decoder_head_mask = decoder_head_mask,
            cross_attn_head_mask = cross_attn_head_mask,
            encoder_outputs = encoder_outputs,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds if encoder_outputs is None else None,
            decoder_inputs_embeds = decoder_inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )

        return outputs

    def input_to_encoder_outputs(
        self,
        input_frames: Optional[torch.LongTensor] = None,
        encoder_prompt: Optional[torch.LongTensor] = None,
        encoder_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        **Encodes the multimodal input and returns encoder outputs.**

        This method processes multimodal inputs (video frames, text, and embeddings) 
        to obtain `encoder_outputs`. It is primarily used during `model.generate()` 
        to retrieve encoder representations before decoding.

        ### **Args:**
        - `input_frames` (Optional[torch.LongTensor], shape: `(B, N_frames, C, W, H)`):  
        The batch of video input frames, where:
            - `B` = batch size  
            - `N_frames` = number of frames per sample  
            - `C` = number of channels  
            - `W` = frame width  
            - `H` = frame height  

        - `encoder_prompt` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
        A prompt consisting of tokenized text that is prepended to the model's input.

        - `encoder_prompt_length_padding_mask` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
        Mask indicating padding tokens in the encoder prompt.

        - `input_ids` (Optional[torch.Tensor], shape: `(B, S_text)`):  
        Tokenized text input IDs.  
        - If `None`, the model relies on `input_frames` for input embeddings.

        - `attention_mask` (Optional[torch.Tensor], shape: `(B, N_frames)`):  
        A mask indicating which frames should be attended to (`1`) and which should be ignored (`0`).

        - `head_mask` (Optional[torch.Tensor], shape: `(num_layers, num_heads)`):  
        Mask for attention heads in the encoder.

        - `inputs_embeds` (Optional[torch.Tensor], shape: `(B, S_text, hidden_dim)`):  
        Precomputed input embeddings instead of `input_ids`.  
        - If `None`, embeddings are computed from `input_frames`.

        - `output_attentions` (Optional[bool], default=`None`):  
        If `True`, the model returns attention weights.

        - `output_hidden_states` (Optional[bool], default=`None`):  
        If `True`, the model returns hidden states of all layers.

        - `return_dict` (Optional[bool], default=`None`):  
        If `True`, returns a `BaseModelOutput` instead of a tuple.

        ### **Returns:**
        - `BaseModelOutput` or `Tuple`:  
        The encoder outputs containing:
            - `last_hidden_state` (torch.FloatTensor, shape `(B, S_text, hidden_dim)`) → Final encoder hidden states.
            - `hidden_states` (Optional[Tuple[torch.FloatTensor]]) → Hidden states from all layers (if `output_hidden_states=True`).
            - `attentions` (Optional[Tuple[torch.FloatTensor]]) → Attention scores (if `output_attentions=True`).

        ### **Processing Steps:**
        1. **Compute Input Embeddings:**  
        - If `inputs_embeds` is not provided, extract features using `feature_extractor(input_frames)`.
        - If a Multimodal Mapper (`multimodal_mapper`) is available, apply it to the embeddings.

        2. **Merge Modalities:**  
        - Combine `inputs_embeds` with `encoder_prompt`, if available.
        - Use `merge_modalities()` to align visual and text inputs before passing them to the encoder.

        3. **Encode Input Representations:**  
        - The processed embeddings are passed to the Transformer encoder to generate `encoder_outputs`.

        ### **Example Usage:**
        ```python
        model = MultiModalEmbedderModel(config)
        input_frames = torch.randn(2, 16, 3, 224, 224)  # Batch of 2 videos
        encoder_prompt = torch.randint(0, 50265, (2, 5))  # Random tokenized prompt

        encoder_outputs = model.input_to_encoder_outputs(input_frames=input_frames, encoder_prompt=encoder_prompt)
        print(encoder_outputs.last_hidden_state.shape)  # Output: (2, sequence_length, hidden_dim)
        ```
        """
        if inputs_embeds is None and input_frames is not None:
            if self.feature_extractor is None:
                inputs_embeds = input_frames
            else:
                inputs_embeds = self.feature_extractor(input_frames)
            
        if self.multimodal_mapper is not None and inputs_embeds is not None:
            inputs_embeds, attention_mask = self.multimodal_mapper(inputs_embeds, attention_mask)

        if inputs_embeds is None:
            inputs_embeds = self.get_backbone_encoder.embed_tokens(input_ids)
            input_ids = None

        inputs_embeds, attention_mask = merge_modalities(
            x=inputs_embeds, 
            padding_mask=attention_mask, 
            prompt=encoder_prompt, 
            prompt_length_padding_mask=encoder_prompt_length_padding_mask,
            embeddings_module=self.get_backbone_encoder.embed_tokens, 
            pad_idx=self.pad_token_id, 
            eos_idx=self.eos_token_id, 
        )

        return self.get_backbone_encoder(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        **Prepares model inputs for autoregressive text generation.**

        This method adapts the inputs before passing them to the `backbone` model 
        during text generation (e.g., beam search or greedy decoding). It ensures 
        stability by handling empty `past_key_values` and properly structuring the 
        inputs for multimodal generation.

        ### **Args:**
        - `*args`: Positional arguments passed to the backbone model.
        - `**kwargs`: Keyword arguments containing:
            - `past_key_values` (Optional[Tuple[Tuple[torch.FloatTensor]]]):  
            Cached key-value states from previous decoding steps.  
            - If empty (`()`), it is set to `None` to prevent errors in final autoregression steps.
            - `input_frames` (Optional[torch.LongTensor], shape: `(B, N_frames, C, W, H)`):  
            Video input frames.
            - `inputs_embeds` (Optional[torch.Tensor], shape: `(B, S_text, hidden_dim)`):  
            Precomputed input embeddings instead of `input_ids`.
            - `encoder_prompt` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
            Prompt prepended to the input sequence.
            - `encoder_prompt_length_padding_mask` (Optional[torch.LongTensor], shape: `(B, prompt_n_tokens)`):  
            Padding mask for the encoder prompt.

        ### **Returns:**
        - `dict`: A dictionary containing all required inputs for the `backbone.generate()` function.

        ### **Processing Steps:**
        1. **Handle Empty `past_key_values`:**  
        - If `past_key_values` is an empty tuple (`()`), it is replaced with `None`.
        
        2. **Retrieve Backbone Model Inputs:**  
        - Calls `self.backbone.prepare_inputs_for_generation(*args, **kwargs)` to get base model inputs.
        
        3. **Add Multimodal Inputs:**  
        - If `input_frames`, `inputs_embeds`, `encoder_prompt`, or `encoder_prompt_length_padding_mask` 
            are present in `kwargs`, they are added to the model input dictionary.

        ### **Example Usage:**
        ```python
        model = MultiModalEmbedderModel(config)
        input_frames = torch.randn(2, 16, 3, 224, 224)
        past_key_values = None  # First decoding step

        model_inputs = model.prepare_inputs_for_generation(
            past_key_values=past_key_values, input_frames=input_frames
        )
        print(model_inputs.keys())  # Output: dict_keys(['input_frames', 'past_key_values'])
        ```
        """

        if kwargs.get('past_key_values', ()) == ():
            kwargs['past_key_values'] = None

        model_inputs = self.backbone.prepare_inputs_for_generation(*args, **kwargs)

        input_frames = kwargs.get('input_frames', None)
        if input_frames is not None:
            model_inputs["input_frames"] = input_frames

        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds

        encoder_prompt = kwargs.get('encoder_prompt', None)
        if encoder_prompt is not None:
            model_inputs["encoder_prompt"] = encoder_prompt

        encoder_prompt_length_padding_mask = kwargs.get('encoder_prompt_length_padding_mask', None)
        if encoder_prompt_length_padding_mask is not None:
            model_inputs["encoder_prompt_length_padding_mask"] = encoder_prompt_length_padding_mask

        return model_inputs

    def get_encoder(self):
        """
        **Retrieves the encoder component of the model.**

        This method returns an `EncoderWrapper`, which encapsulates the model’s encoder 
        for use in downstream tasks like sequence-to-sequence generation.

        ### **Returns:**
        - `EncoderWrapper`: The encoder module of the model.
        """
        return EncoderWrapper(self)
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """
        **Reorders the past key-value cache for beam search decoding.**

        During beam search, this method reorders `past_key_values` based on the 
        surviving beams (`beam_idx`), ensuring that cached values remain aligned 
        with the correct sequences.

        ### **Args:**
        - `past_key_values` (Tuple[Tuple[torch.FloatTensor]]):  
        Cached self-attention and cross-attention key-value pairs from previous decoding steps.
        - `beam_idx` (torch.LongTensor, shape `(num_beams,)`):  
        The indices of the beams that survived the last decoding step.

        ### **Returns:**
        - `Tuple[Tuple[torch.FloatTensor]]`: The reordered past key-value states.
        """
        return self.backbone._reorder_cache(past_key_values, beam_idx)
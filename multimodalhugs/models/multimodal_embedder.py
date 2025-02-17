import logging
import math
import importlib

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

from ruamel.yaml import YAML
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union

from multimodalhugs.models import EncoderWrapper, freeze_module_parameters
from multimodalhugs.models.registry import register_model
from multimodalhugs.modules import VLMapper, FeatureExtractor, SpecialTokensEmbeddings, get_feature_extractor_class
from multimodalhugs.utils import serialize_config
logger = logging.getLogger(__name__)

def init_encoder_new_embeddings(cfg, new_embeddings, pretrained_embeddings, tokenizer):

    if cfg.init_lang_abbr is not None and cfg.init_lang_abbr!="avg":
        lang_idx = "__" + cfg.init_lang_abbr + "__"
        lang_idx = tokenizer.convert_tokens_to_ids(lang_idx)
        new_embeddings.weight.data = pretrained_embeddings.weight.data[lang_idx].expand_as(new_embeddings.weight).clone()
        logger.info(f"Language embedding layer initialized with weights from the '{cfg.init_lang_abbr}' language.")
    
    elif cfg.init_lang_abbr=="avg": # We iniciclize the weights from the average weight from all the MT language tokens
        lang_idx = tokenizer.special_tokens_map['additional_special_tokens']
        lang_idx = [tokenizer.convert_tokens_to_ids(lang_idx_) for lang_idx_ in lang_idx]

        avg_tensor = []
        
        for lang_idx_ in lang_idx:
            avg_tensor.append(pretrained_embeddings.weight.data[lang_idx_])
        if len(avg_tensor) > 0:
            avg_tensor = torch.stack(avg_tensor).mean(dim=0)
            new_embeddings.weight.data = avg_tensor.expand_as(new_embeddings.weight).clone()
            logger.info(f"Language embedding layer initialized with weights from the average language embeddings.")
    else:
        logger.info("Language embedding layer initialized from scratch as no initial language was provided.")
    return new_embeddings
    
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
    using Hugging Face's MODEL_WITH_LM_HEAD_MAPPING_NAMES mapping.

    Args:
        model_type (str): The model type (e.g., 'bert', 't5', etc.).

    Returns:
        The corresponding model class.

    Raises:
        ValueError: If `model_type` is not found in MODEL_WITH_LM_HEAD_MAPPING_NAMES.
        ImportError: If the module or model class cannot be imported.
    """
    if model_type not in MODEL_WITH_LM_HEAD_MAPPING_NAMES:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available options: {list(MODEL_WITH_LM_HEAD_MAPPING_NAMES.keys())}"
        )
    
    model_class_name = MODEL_WITH_LM_HEAD_MAPPING_NAMES[model_type]
    # Assumes that the module name corresponds to model_type, replacing dashes with underscores
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


@dataclass
class MultiModalEmbedderConfig(PretrainedConfig):
    model_type: str = field(
        default="multimodal_embedder", metadata={"help": "Name of the model to be used."}
    )
    feat_dim: int = field(
        default=512, metadata={"help": "Dimention of the Feature Extractor output."}
    )
    feature_extractor_type: Optional[str] = field(
        default=None, metadata={"help": "Feature Extractor type to be used."}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    pretrained_feature_extractor: Optional[str] = field(
        default=None, metadata={"help": "Pretrained Feature Extractor or path to the Pretrained Feature Extractor checkpoint."}
    )
    freeze_feature_extractor: bool = field(
        default=False, metadata={"help": "if True, the feature_extractor parameters are frozen during training."}
    )
    vl_mapper_type: str = field(
        default="linear", metadata={"help": "Chose the VL Mapper type. Options: 'linear', 'adapter'."}
    )
    vl_mapper_layer_norm_before: bool = field(
        default=False, metadata={"help": "if True, adds a LayerNorm before the vl_mapper"}
    )
    vl_mapper_layer_norm: bool = field(
        default=False, metadata={"help": "if True, adds a LayerNorm inside the vl_mapper"}
    )
    vl_mapper_activation: bool = field(
        default=False, metadata={"help": "if True, applies a ReLu at the vl_mapper output"}
    )
    vl_factor: Optional[int] = field(
        default=None,
        metadata={"help": "If specified, use an adapter as V-L mapper whose overparameterization is given by the given factor"}
    )
    vl_mapper_dropout: Optional[float] = field(
        default=None, metadata={"help": "Dropout probabilty for the vl_mapper"}
    )
    freeze_vl_mapper: bool = field(
        default=False, metadata={"help": "if True, the vl_mapper parameters are frozen during training."}
    )
    new_embeddings_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "vocab_size of the source language embeddings"}
    )
    backbone_used_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "Original vocab_size of the backbone excluding garbage embeddings"}
    )
    init_lang_abbr: Optional[str] = field(
        default=None,
        metadata={
            "help": "Language abbreviation of the language you want to use as initialization of the embeddings layer of the new languages. If the value is None, the embeddings layer will be initialized from scratch."
        }
    )
    freeze_new_embeddings: bool = field(
        default=False, metadata={"help": "if True, the new_embeddings parameters are frozen during training."}
    )
    freeze_old_embeddings: bool = field(
        default=False, metadata={"help": "if True, the new_embeddings parameters are frozen during training."}
    )
    backbone_name: str = field(
        default="m2m100", metadata={"help": "Name of the model to be used as a backbone"}
    )
    backbone_config: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Hyperparameters in case the backbone is inicialized from scratch."}
    )
    pretrained_backbone: Optional[str] = field(
        default=None, metadata={"help": "Pretrained Backbone or path to the Pretrained Backbone checkpoint."}
    )
    freeze_backbone: bool = field(
        default=False, metadata={"help": "if True, the backbone parameters are frozen during training."}
    )
    d_model: Optional[int] = field(
        default=None, metadata={"help": "Dimention of the model"}
    )
    feature_extractor_cfg: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Hyperparameters in case the feature_extractor is inicialized from scratch."}
    )
    is_encoder_decoder: bool = field(
        default=True,
    )
    pad_token_id: Optional[int] = field(
        default=None, metadata={"help": "Allows to specify the vocabulary index of the <pad> token to be added to the multimodal sequences."}
    )
    bos_token_id: Optional[int] = field(
        default=None, metadata={"help": "Allows to specify the vocabulary index of the <bos> token to be added to the multimodal sequences."}
    )
    eos_token_id: Optional[int] = field(
        default=None, metadata={"help": "Allows to specify the vocabulary index of the <eos> token to be added to the multimodal sequences."}
    )
    max_length: int = field(
        default=1024, metadata={"help": "The maximum target length to use when predicting with the generate method."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    setattr(self, key, value)

        if kwargs and 'backbone_name' in kwargs:
            backbone_name = kwargs['backbone_name']
            BackboneConfigClass = get_backbone_config_class(backbone_name)
            self.backbone_config = BackboneConfigClass(**kwargs.get('backbone_config', {}))
        else:
            self.backbone_config = None

        if kwargs and 'feature_extractor_type' in kwargs:
            feature_extractor_type = kwargs['feature_extractor_type']
            FeatureExtractorConfigClass = get_feature_extractor_class(feature_extractor_type)[1]
            self.feature_extractor_config = FeatureExtractorConfigClass(**kwargs.get('feature_extractor_cfg', {}))
        else:
            self.feature_extractor_config = None

        self.is_encoder_decoder = True


# Define the custom model class
@register_model("multimodal_embedder")
class MultiModalEmbedderModel(PreTrainedModel):
    config_class = MultiModalEmbedderConfig
    base_model_prefix = "multimodal_embedder"

    def __init__(self, config):
        super().__init__(config)
        
        # Feature Extractor
        self.feature_extractor = FeatureExtractor(
            feature_extractor_type=config.feature_extractor_type, 
            pretrained_module=config.pretrained_feature_extractor, 
            config=config.feature_extractor_config, 
        ) if config.feature_extractor_type is not None else None

        if config.freeze_feature_extractor and self.feature_extractor is not None:
            freeze_module_parameters(self.feature_extractor)
            
        # VL Mapper
        self.vl_mapper = VLMapper(
            feat_dim=config.feat_dim, 
            output_dim=config.d_model, 
            mapping_layer_type=config.vl_mapper_type, 
            layer_norm_before=config.vl_mapper_layer_norm_before,
            adapter_factor=config.vl_factor, 
            p_dropout=config.vl_mapper_dropout, 
            layer_norm=config.vl_mapper_layer_norm, 
            activation=config.vl_mapper_activation,
        )
        if config.freeze_vl_mapper:
            freeze_module_parameters(self.vl_mapper)

        # Lang Embedings
        self.special_tokens_embeddings = SpecialTokensEmbeddings(
            old_vocab_size=config.backbone_used_vocab_size if config.backbone_used_vocab_size is not None else config.backbone_config.vocab_size,
            new_vocab_size=config.new_embeddings_vocab_size if config.new_embeddings_vocab_size is not None else 0,
            embed_dim=config.d_model,
            scale_embeddings=False if config.no_scale_embedding else True,
            pad_idx=config.pad_token_id,
            eos_idx=config.eos_token_id,
        )

        if config.freeze_new_embeddings:
            if self.special_tokens_embeddings.special_tokens_embeddings.new_embeddings is not None:
                freeze_module_parameters(self.special_tokens_embeddings.special_tokens_embeddings.new_embeddings)
        if config.freeze_old_embeddings:
            freeze_module_parameters(self.special_tokens_embeddings.special_tokens_embeddings.old_embeddings)

        # Backbone
        BackboneModelClass = get_backbone_model_class(config.backbone_name)

        if config.backbone_config is not None:
            self.backbone = BackboneModelClass(config.backbone_config)
        else:
            self.backbone = BackboneModelClass.from_pretrained(config.pretrained_backbone)
            
        if config.freeze_backbone:
            freeze_module_parameters(self.backbone)

        # Others
        self.max_length = config.max_length
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(config.d_model)
        encoder = self.backbone.encoder if hasattr(self.backbone, 'encoder') else self.backbone.model.encoder
        self.padding_token = encoder.embed_tokens(torch.tensor([config.pad_token_id], dtype=torch.long, device=self.backbone.device)).detach().numpy() if config.pad_token_id is not None else config.pad_token_id
        self.eos_token = encoder.embed_tokens(torch.tensor([config.eos_token_id], dtype=torch.long, device=self.backbone.device)).detach().numpy() if config.eos_token_id is not None else config.eos_token_id

    def get_input_embeddings(self):
        if hasattr(self.backbone, 'shared'):
            return self.backbone.shared
        elif hasattr(self.backbone, 'model'):
            if hasattr(self.backbone.model, 'shared'):
                return getattr(self.backbone.model, 'shared', None) 
        else:
            return None


    def set_input_embeddings(self, value):
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
        return self.backbone.lm_head

    @property
    def lm_head(self):
        if hasattr(self.backbone, 'lm_head'):
            return self.backbone.lm_head
        return None

    @property
    def language_encoder(self):
        return self.backbone.encoder if hasattr(self.backbone, 'encoder') else self.backbone.model.encoder

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
        BackboneModelClass = get_backbone_model_class(cfg.backbone_name)
        if cfg.pretrained_backbone is not None:
            backbone = BackboneModelClass.from_pretrained(cfg.pretrained_backbone)
            cfg.backbone_config = AutoConfig.from_pretrained(cfg.pretrained_backbone)
        else:
            backbone = BackboneModelClass(cfg.backbone_config)
        
        cfg.d_model = cfg.d_model or cfg.backbone_config.d_model

        print(f"backbone: \n{backbone}")

        # Determine EOS and PAD token indices
        pad_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.pad_token)
        bos_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.bos_token)
        eos_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.eos_token)

        # Update configuration with these values if not already defined
        cfg.pad_token_id = cfg.pad_token_id or pad_token_id
        cfg.bos_token_id = cfg.bos_token_id or bos_token_id
        cfg.eos_token_id = cfg.eos_token_id or eos_token_id
        tokenizer_vocab_size = getattr(src_tokenizer, "total_vocab_size", src_tokenizer.vocab_size)
        cfg.backbone_used_vocab_size = cfg.backbone_used_vocab_size or (tokenizer_vocab_size - len(new_vocab_tokens))
        cfg.new_embeddings_vocab_size = cfg.new_embeddings_vocab_size or len(new_vocab_tokens)

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
            config_data['model']['backbone_used_vocab_size'] = cfg.backbone_used_vocab_size
            config_data['model']['new_embeddings_vocab_size'] = cfg.new_embeddings_vocab_size
            
            # Save the updated configuration back to the file
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file)

        # Handling pretrained and language-specific embeddings
        pretrained_embeddings = backbone.encoder.embed_tokens if hasattr(backbone, 'encoder') else backbone.model.encoder.embed_tokens # torch.Size([128112, 1024])

        source_embeddings = SpecialTokensEmbeddings.build_module(
            old_vocab_size=cfg.backbone_used_vocab_size, 
            new_vocab_size=len(new_vocab_tokens), 
            embed_dim=cfg.d_model,
            scale_embeddings=False if cfg.no_scale_embedding else True,
            pad_idx=cfg.pad_token_id,
            eos_idx=cfg.eos_token_id,
            old_embs_weight=pretrained_embeddings.weight, 
        )
        
        source_embeddings.special_tokens_embeddings.new_embeddings = init_encoder_new_embeddings(
            cfg=cfg, 
            new_embeddings=source_embeddings.special_tokens_embeddings.new_embeddings, 
            pretrained_embeddings=source_embeddings.special_tokens_embeddings.old_embeddings, 
            tokenizer=tgt_tokenizer
        )

        # Create an instance of the model
        model = cls(config=cfg)

        model.backbone.load_state_dict(backbone.state_dict()) # copy the weights from the backbone instance to the model.backbone 
        model.special_tokens_embeddings.special_tokens_embeddings.new_embeddings.weight.data = source_embeddings.special_tokens_embeddings.new_embeddings.weight.data.clone()
        model.special_tokens_embeddings.special_tokens_embeddings.old_embeddings.weight.data = source_embeddings.special_tokens_embeddings.old_embeddings.weight.data.clone()

        # Converts all tensors in the model to contiguous
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        return model

    def forward(
        self,
        input_frames: Optional[torch.LongTensor] = None,
        src_prompt: Optional[torch.LongTensor] = None,
        source_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
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
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.

        INPUTS:
            - input_ids (Text2Text): B x S_text <— Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
            - input_frames (also known as 'src_tokens'): B x N_frames x C x W x H <- Multimodal input minibatch
            - src_prompt: B x prompt_n_tokens <- prompt to be added to 'inputs_embeds' before fitting them to the Language Model
            - source_prompt_length_padding_mask: B x prompt_n_tokens <- source prompt padding_mask.
            - attention_mask (also known as 'encoder_padding_mask'): B x N_frames <- 0 indicates padding elements
            - decoder_input_ids (also known as 'source_text'): B x T_text <- Should look as ['</s>', '<tgt_lang>', '<token_a>', '<token_b>', '<token_c>'] if teacher forcing, otherwise None. In Generation: ['<s>', '<tgt_lang>']
            - labels (also known as 'source_text'): B x T_text <- Just needed in training. Should look as: ['<tgt_lang>', '<token_a>', '<token_b>', '<token_c>', '</s>']
            - decoder_attention_mask: B x T_text <- 0 indicates padding elements
        """
        if inputs_embeds is None:
            inputs_embeds = self.feature_extractor(input_frames)

        if self.vl_mapper is not None:
            inputs_embeds = self.vl_mapper(inputs_embeds)

        inputs_embeds, attention_mask =  self.special_tokens_embeddings(inputs_embeds, attention_mask, src_prompt, source_prompt_length_padding_mask)

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
        src_prompt: Optional[torch.LongTensor] = None,
        source_prompt_length_padding_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        INPUTS:
            - input_ids (Text2Text): B x S_text <— Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
            - input_frames (also known as 'src_tokens'): B x N_frames x C x W x H <- Multimodal input minibatch
            - src_prompt: B x prompt_n_tokens <- Language tokens to be added to 'inputs_embeds' before fitting them to the Language Model
            - source_prompt_length_padding_mask: B x prompt_n_tokens <- source prompt padding_mask.
            - attention_mask (also known as 'encoder_padding_mask'): B x N_frames <- 0 indicates padding elements
        """
        
        if inputs_embeds is None:
            inputs_embeds = self.feature_extractor(input_frames)
        if self.vl_mapper is not None:
            inputs_embeds = self.vl_mapper(inputs_embeds)
        inputs_embeds, attention_mask =  self.special_tokens_embeddings(inputs_embeds, attention_mask, src_prompt, source_prompt_length_padding_mask)

        return self.language_encoder(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Added a condition to handle empty `past_key_values` which occurred unpredictably in the final autoregression step, causing errors during generation. This ensures stability across all steps.
        if kwargs.get('past_key_values', ()) == ():
            kwargs['past_key_values'] = None

        model_inputs = self.backbone.prepare_inputs_for_generation(*args, **kwargs)

        input_frames = kwargs.get('input_frames', None)
        if input_frames is not None:
            model_inputs["input_frames"] = input_frames

        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds

        src_prompt = kwargs.get('src_prompt', None)
        if src_prompt is not None:
            model_inputs["src_prompt"] = src_prompt

        source_prompt_length_padding_mask = kwargs.get('source_prompt_length_padding_mask', None)
        if source_prompt_length_padding_mask is not None:
            model_inputs["source_prompt_length_padding_mask"] = source_prompt_length_padding_mask

        return model_inputs

    def get_encoder(self):
        return EncoderWrapper(self)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.backbone._reorder_cache(past_key_values, beam_idx)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return M2M100ForConditionalGeneration._reorder_cache(past_key_values, beam_idx)
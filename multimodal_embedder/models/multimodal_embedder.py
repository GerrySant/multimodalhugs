# Standard libraries
import logging
import math

# Third-party libraries
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from transformers import (
    BertConfig, BertModel, CLIPConfig, CLIPModel, M2M100Config, M2M100ForConditionalGeneration, 
    PreTrainedModel, PretrainedConfig
)
from transformers.modeling_outputs import Seq2SeqLMOutput

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union

# Local application libraries
from multimodal_embedder.models import EncoderWrapper, freeze_module_parameters
from multimodal_embedder.modules import VLMapper, FeatureExtractor, get_feature_extractor_class
from multimodal_embedder.utils import serialize_config

logger = logging.getLogger(__name__)

def init_encoder_lang_embeddings(cfg, lang_embeddings, pretrained_embeddings, tokenizer):

    if cfg.init_lang_abbr is not None and cfg.init_lang_abbr!="avg":
        lang_idx = "__" + cfg.init_lang_abbr + "__"
        lang_idx = tokenizer.convert_tokens_to_ids(lang_idx)
        lang_embeddings.weight.data = pretrained_embeddings.weight.data[lang_idx].expand_as(lang_embeddings.weight).clone()
        logger.info(f"Language embedding layer initialized with weights from the '{cfg.init_lang_abbr}' language.")
    
    elif cfg.init_lang_abbr=="avg": # We iniciclize the weights from the average weight from all the MT language tokens
        lang_idx = tokenizer.special_tokens_map['additional_special_tokens']
        lang_idx = [tokenizer.convert_tokens_to_ids(lang_idx_) for lang_idx_ in lang_idx]

        avg_tensor = []
        for lang_idx_ in lang_idx:
            avg_tensor.append(pretrained_embeddings.weight.data[lang_idx_])
        if len(avg_tensor) > 0:
            avg_tensor = torch.stack(avg_tensor).mean(dim=0)
            lang_embeddings.weight.data = avg_tensor.expand_as(lang_embeddings.weight).clone()
            logger.info(f"Language embedding layer initialized with weights from the average language embeddings.")
    else:
        logger.info("Language embedding layer initialized from scratch as no initial language was provided.")
    # Ensure that the special tokens (excluding Languages) of the new language embedding have the same weight from the original embedding layer
    special_tokens = [tokenizer.special_tokens_map[key] for key in tokenizer.special_tokens_map.keys() if key != "additional_special_tokens"]
    special_tokens = [tokenizer.convert_tokens_to_ids(special_token) for special_token in special_tokens]
    for token_index in special_tokens:
        lang_embeddings.weight.data[token_index] = pretrained_embeddings.weight.data[token_index]
    return lang_embeddings

# Factory function to create the appropriate configuration class based on backbone_name
def get_backbone_config_class(backbone_name):
    if backbone_name == "m2m100": # The actual version only supports M2M as backbone
        return M2M100Config
    else:
        raise ValueError(f"Unknown backbone name: {backbone_name}")

# Factory function to create the appropriate model class based on backbone_name
def get_backbone_model_class(backbone_name):
    if backbone_name == "m2m100": # The actual version only supports M2M as backbone
        return M2M100ForConditionalGeneration
    else:
        raise ValueError(f"Unknown backbone name: {backbone_name}")

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
    lang_embeddings_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "vocab_size of the source language embeddings"}
    )
    init_lang_abbr: Optional[str] = field(
        default=None,
        metadata={
            "help": "Language abbreviation of the language you want to use as initialization of the embeddings layer of the new languages. If the value is None, the embeddings layer will be initialized from scratch."
        }
    )
    freeze_lang_embeddings: bool = field(
        default=False, metadata={"help": "if True, the lang_embeddings parameters are frozen during training."}
    )
    backbone_name: str = field(
        default="m2m100", metadata={"help": "Name of the model to be used as a backbone"}
    )
    backbone_cfg: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Hyperparameters in case the backbone is inicialized from scratch."}
    )
    pretrained_backbone: Optional[str] = field(
        default=None, metadata={"help": "Pretrained Backbone or path to the Pretrained Backbone checkpoint."}
    )
    freeze_backbone: bool = field(
        default=False, metadata={"help": "if True, the backbone parameters are frozen during training."}
    )
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "Dimention of the encoder backbone"}
    )
    feature_extractor_cfg: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Hyperparameters in case the feature_extractor is inicialized from scratch."}
    )
    is_encoder_decoder: bool = field(
        default=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs and 'backbone_name' in kwargs:
            backbone_name = kwargs['backbone_name']
            BackboneConfigClass = get_backbone_config_class(backbone_name)
            self.backbone_config = BackboneConfigClass(**kwargs.get('backbone_cfg', {}))
        else:
            self.backbone_config = None

        if kwargs and 'feature_extractor_type' in kwargs:
            feature_extractor_type = kwargs['feature_extractor_type']
            FeatureExtractorConfigClass = get_feature_extractor_class(feature_extractor_type)[1]
            self.feature_extractor_config = FeatureExtractorConfigClass(**kwargs.get('feature_extractor_cfg', {}))
        else:
            self.feature_extractor_config = None
        
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    setattr(self, key, value)
        self.is_encoder_decoder = True


# Define the custom model class
class MultiModalEmbedderModel(PreTrainedModel):
    config_class = MultiModalEmbedderConfig
    base_model_prefix = "multimodal_embedder"

    def __init__(self, config, lang_embeddings = None, eos_indx = None, pad_index = None):
        super().__init__(config)
        
        # Feature Extractor
        self.feature_extractor = FeatureExtractor(
            feature_extractor_type=config.feature_extractor_type, 
            pretrained_module=config.pretrained_feature_extractor, 
            config=config.feature_extractor_config, 
        )
        if config.freeze_feature_extractor:
            freeze_module_parameters(self.feature_extractor)
            
        # VL Mapper
        self.vl_mapper = VLMapper(
            feat_dim=config.feat_dim, 
            output_dim=config.encoder_embed_dim, 
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
        if lang_embeddings is not None:
            self.lang_embeddings = lang_embeddings
        else:
            self.lang_embeddings = nn.Embedding(num_embeddings=config.lang_embeddings_vocab_size, embedding_dim=config.encoder_embed_dim)
        if config.freeze_lang_embeddings:
            freeze_module_parameters(self.lang_embeddings)

        # Backbone
        BackboneModelClass = get_backbone_model_class(config.backbone_name)
        if config.pretrained_backbone is not None:
            self.backbone = BackboneModelClass.from_pretrained(config.pretrained_backbone)
        else:
            self.backbone = BackboneModelClass(config.backbone_config)
            
        if config.freeze_backbone:
            freeze_module_parameters(self.backbone)

        # Others
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(config.encoder_embed_dim)
        encoder = self.backbone.encoder if hasattr(self.backbone, 'encoder') else self.backbone.model.encoder
        self.padding_token = encoder.embed_tokens(torch.tensor([pad_index], dtype=torch.long, device=self.backbone.device)).detach().numpy() if pad_index is not None else pad_index
        self.eos_token = encoder.embed_tokens(torch.tensor([eos_indx], dtype=torch.long, device=self.backbone.device)).detach().numpy() if pad_index is not None else eos_indx

    def get_input_embeddings(self):
        return self.backbone.model.shared

    def set_input_embeddings(self, value):
        self.backbone.model.shared = value
        self.backbone.model.encoder.embed_tokens = self.backbone.model.shared
        self.backbone.model.decoder.embed_tokens = self.backbone.model.shared

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
    def build_model(cls, cfg, dataset=None, src_tokenizer = None, tgt_tokenizer = None):
        """Build the MultiModal Embedder model using specific model configuration and dataset."""

        if dataset is not None:
            src_tokenizer = dataset.src_tokenizer
            tgt_tokenizer = dataset.tgt_tokenizer

        if src_tokenizer is None or tgt_tokenizer is None:
            raise ValueError("Please provide the src_tokenizer and the tgt_tokenizer in case the dataset used does not have these as a parameter.")

        if not isinstance(cfg, PretrainedConfig):
            cfg = cls.config_class.from_dict(serialize_config(cfg))
        else:
            cfg = cfg

        BackboneModelClass = get_backbone_model_class(cfg.backbone_name)
        
        if cfg.pretrained_backbone is not None:
            backbone = BackboneModelClass.from_pretrained(cfg.pretrained_backbone)
        else:
            backbone = BackboneModelClass(cfg.backbone_config)

        # Handling pretrained and language-specific embeddings
        pretrained_embeddings = backbone.encoder.embed_tokens if hasattr(backbone, 'encoder') else backbone.model.encoder.embed_tokens

        lang_embeddings = nn.Embedding(num_embeddings=src_tokenizer.vocab_size, embedding_dim=cfg.encoder_embed_dim)

        lang_embeddings = init_encoder_lang_embeddings(
            cfg=cfg, 
            lang_embeddings=lang_embeddings, 
            pretrained_embeddings=pretrained_embeddings, 
            tokenizer=tgt_tokenizer
        )

        # EOS and PAD token indices
        eos_index = src_tokenizer.convert_tokens_to_ids(src_tokenizer.eos_token)
        pad_index = src_tokenizer.convert_tokens_to_ids(src_tokenizer.pad_token)

        # Create an instance of the model
        model = cls(config=cfg, lang_embeddings=lang_embeddings, eos_indx=eos_index, pad_index=pad_index)
        return model
 
        
    def forward_special_tokens(self, x, encoder_padding_mask, src_langtoks):
        """
        It adds and/or corrects the special tokens from the input secuence:
            # '<src_lang>', ...,  '</s>', '<pad>', '<pad>'
        
        INPUTS:
            - x: B x N_tokens x Embed_dim
            - encoder_padding_mask: B x N_tokens <- 0 indicates padding elements
            - src_langtoks: B x 1     
        """
        # Append <src_lang>:
        if src_langtoks is not None:
            src_langtoks = self.lang_embeddings(src_langtoks)
            x = torch.cat((src_langtoks, x), dim=1)

            # Correct Padding Mask
            new_mask_entry = torch.full((encoder_padding_mask.size(0), 1), 1, dtype=encoder_padding_mask.dtype, device=encoder_padding_mask.device)
            encoder_padding_mask = torch.cat([new_mask_entry, encoder_padding_mask], dim=1)
        
        # Adjust <pad> tokens and add <eos> token to every secuence in the batch:
        if self.padding_token is not None and self.eos_token is not None:
            # Adjust <pad> tokens according the exped ones by the pretrained LM.
            bool_padding_mask = encoder_padding_mask == 0
            
            x[bool_padding_mask] = torch.from_numpy(self.padding_token).to(encoder_padding_mask.device)
            # Add <eos> token to every secuence in the batch
        
            # Adjust Padding Mask to reflect the addition of the <eos> token
            new_mask_entry = torch.full((encoder_padding_mask.size(0), 1), 1, dtype=encoder_padding_mask.dtype, device=encoder_padding_mask.device)
            encoder_padding_mask = torch.cat([new_mask_entry, encoder_padding_mask], dim=1)
            
            # Create a mask indicating the position where the <eos> vector should be inserted in each minibatch sequence.
            eos_inster_mask = torch.zeros_like(encoder_padding_mask)
            last_indices = encoder_padding_mask.size(1) - torch.argmax(encoder_padding_mask.flip(dims=[1]), dim=1) - 1
            rows = torch.arange(encoder_padding_mask.size(0))
            eos_inster_mask[rows, last_indices] = 1
            eos_inster_mask = eos_inster_mask != 0
            
            # Add a padding token to each of the minibatch sequences to prepare it for the allocation of the <eos> tokens. Then append them.
            new_padding_column = torch.from_numpy(self.padding_token).to(encoder_padding_mask.device).repeat(x.size(0), 1).unsqueeze(1)
            x = torch.cat([x, new_padding_column], dim=1)
            x[eos_inster_mask] = torch.from_numpy(self.eos_token).to(encoder_padding_mask.device)
        x = self.embed_scale * x
            
        return x, encoder_padding_mask

    def forward(
        self,
        input_frames: Optional[torch.LongTensor] = None,
        src_langtoks: Optional[torch.LongTensor] = None,
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
            - src_langtoks: B x 1 <- Language tokens to be added to 'inputs_embeds' before fitting them to the Language Model
            - attention_mask (also known as 'encoder_padding_mask'): B x N_frames <- 0 indicates padding elements
            - decoder_input_ids (also known as 'source_text'): B x T_text <- Should look as ['</s>', '<tgt_lang>', '<token_a>', '<token_b>', '<token_c>'] if teacher forcing, otherwise None. In Generation: ['<s>', '<tgt_lang>']
            - labels (also known as 'source_text'): B x T_text <- Just needed in training. Should look as: ['<tgt_lang>', '<token_a>', '<token_b>', '<token_c>', '</s>']
            - decoder_attention_mask: B x T_text <- 0 indicates padding elements
        """

        if inputs_embeds is None:
            inputs_embeds = self.feature_extractor(input_frames)
            if self.vl_mapper is not None:
                inputs_embeds = self.vl_mapper(inputs_embeds)
        inputs_embeds, attention_mask =  self.forward_special_tokens(inputs_embeds, attention_mask, src_langtoks)

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
        src_langtoks: Optional[torch.LongTensor] = None,
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
            - src_langtoks: B x 1 <- Language tokens to be added to 'inputs_embeds' before fitting them to the Language Model
            - attention_mask (also known as 'encoder_padding_mask'): B x N_frames <- 0 indicates padding elements
        """
        
        if inputs_embeds is None:
            inputs_embeds = self.feature_extractor(input_frames)
            if self.vl_mapper is not None:
                inputs_embeds = self.vl_mapper(inputs_embeds)
            inputs_embeds, attention_mask =  self.forward_special_tokens(inputs_embeds, attention_mask, src_langtoks)

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
        model_inputs["input_frames"] = kwargs['input_frames']
        model_inputs["src_langtoks"] = kwargs['src_langtoks']
        return model_inputs

    def get_encoder(self):
        return EncoderWrapper(self)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return M2M100ForConditionalGeneration._reorder_cache(past_key_values, beam_idx)
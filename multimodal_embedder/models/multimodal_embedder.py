# Standard libraries
import logging
import math

# Third-party libraries
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import (
    BertConfig, BertModel, CLIPConfig, CLIPModel, M2M100Config, M2M100Model, 
    PreTrainedModel, PretrainedConfig
)
from dataclasses import dataclass, field
from typing import Optional

# Local application libraries
from multimodal_embedder.models import freeze_module_parameters
from multimodal_embedder.modules import VLMapper


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
    if backbone_name == "m2m100":
        return M2M100Config
    elif backbone_name == "bert":
        return BertConfig
    else:
        raise ValueError(f"Unknown backbone name: {backbone_name}")

# Factory function to create the appropriate model class based on backbone_name
def get_backbone_model_class(backbone_name):
    if backbone_name == "m2m100":
        return M2M100Model
    elif backbone_name == "bert":
        return BertModel
    else:
        raise ValueError(f"Unknown backbone name: {backbone_name}")

# Factory function to create the appropriate feature extractor class based on feature_extractor_type
def get_feature_extractor_class(feature_extractor_type):
    if feature_extractor_type == "clip":
        return CLIPModel, CLIPConfig
    else:
        raise ValueError(f"Unknown feature extractor type: {feature_extractor_type}")

@dataclass
class MultiModalEmbedderConfig(PretrainedConfig):
    name: str = field(
        default="multimodal_embedder", metadata={"help": "Name of the model to be used."}
    )
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
    pretrained_backbone: Optional[str] = field(
        default=None, metadata={"help": "Pretrained Backbone or path to the Pretrained Backbone checkpoint."}
    )
    freeze_backbone: bool = field(
        default=False, metadata={"help": "if True, the backbone parameters are frozen during training."}
    )
    freeze_lm_head: bool = field(
        default=False, metadata={"help": "if True, the lm_head parameters are frozen during training."}
    )
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "Dimention of the encoder backbone"}
    )

    def __init__(self, cfg=None, **kwargs):
        super().__init__(**kwargs)
        if cfg and 'backbone_name' in cfg:
            backbone_name = cfg.backbone_name
            BackboneConfigClass = get_backbone_config_class(backbone_name)
            self.backbone_config = BackboneConfigClass(**cfg.get(backbone_name, {}))
        else:
            self.backbone_config = None
        
        if cfg and 'feature_extractor_type' in cfg:
            feature_extractor_type = cfg.feature_extractor_type
            FeatureExtractorConfigClass = get_feature_extractor_class(feature_extractor_type)[1]
            self.feature_extractor_config = FeatureExtractorConfigClass(**cfg.get(feature_extractor_type, {}))
        else:
            self.feature_extractor_config = None
        
        if cfg:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            for key, value in cfg_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    setattr(self, key, value)

# Define the custom model class
class MultiModalEmbedderModel(PreTrainedModel):
    config_class = MultiModalEmbedderConfig
    base_model_prefix = "multimodal_embedder"

    def __init__(self, config, lang_embeddings = None, eos_indx = None, pad_index = None):
        super().__init__(config)
        # Feature Extractor
        self.feature_extractor_type = config.feature_extractor_type
        if config.feature_extractor_type:
            FeatureExtractorClass, _ = get_feature_extractor_class(config.feature_extractor_type)
            if config.pretrained_feature_extractor is not None:
                self.feature_extractor = FeatureExtractorClass.from_pretrained(config.pretrained_feature_extractor)
            else:
                self.feature_extractor = FeatureExtractorClass(config.feature_extractor_config)
                
            if isinstance(self.feature_extractor, CLIPModel):
                self.feature_extractor.text_model = None
                self.feature_extractor.text_projection = None
                
            if config.freeze_feature_extractor:
                freeze_module_parameters(self.feature_extractor)
        else:
            self.feature_extractor = None
            
        # VL Mapper
        self.vl_mapper = VLMapper(
            feat_dim = config.feat_dim, 
            output_dim = config.encoder_embed_dim, 
            mapping_layer_type = config.vl_mapper_type, 
            layer_norm_before = config.vl_mapper_layer_norm_before,
            adapter_factor = config.vl_factor, 
            p_dropout = config.vl_mapper_dropout, 
            layer_norm = config.vl_mapper_layer_norm, 
            activation = config.vl_mapper_activation,
        )
        if config.freeze_vl_mapper:
            freeze_module_parameters(self.vl_mapper)

        # Lang Embedings
        self.lang_embeddings = lang_embeddings
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

        # LM Head
        self.lm_head = nn.Linear(config.encoder_embed_dim, self.backbone.shared.num_embeddings, bias=False)
        self.lm_head.weight = nn.Parameter(self.backbone.shared.weight.clone())
        if config.freeze_lm_head:
            freeze_module_parameters(self.lm_head)

        # Others
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(config.encoder_embed_dim)
        self.eos_indx = eos_indx
        self.pad_index = pad_index

    @classmethod
    def build_model(cls, cfg, dataset):
        """Build the MultiModal Embedder model using specific model configuration and dataset."""

        if not isinstance(cfg, PretrainedConfig):
            cfg = cls.config_class.from_dict(cfg)
        else:
            cfg = cfg
    
        BackboneModelClass = get_backbone_model_class(cfg.backbone_name)

        if cfg.pretrained_backbone is not None:
            backbone = BackboneModelClass.from_pretrained(cfg.pretrained_backbone)
        else:
            backbone = BackboneModelClass(cfg.backbone_config)

        # Handling pretrained and language-specific embeddings
        pretrained_embeddings = backbone.encoder.embed_tokens
        lang_embeddings = nn.Embedding(num_embeddings=dataset.src_tokenizer.vocab_size, embedding_dim=cfg.encoder_embed_dim)

        lang_embeddings = init_encoder_lang_embeddings(
            cfg=cfg, 
            lang_embeddings=lang_embeddings, 
            pretrained_embeddings=pretrained_embeddings, 
            tokenizer=dataset.tgt_tokenizer
        )

        # EOS and PAD token indices
        eos_indx = dataset.src_tokenizer.convert_tokens_to_ids(dataset.src_tokenizer.eos_token)
        pad_index = dataset.src_tokenizer.convert_tokens_to_ids(dataset.src_tokenizer.pad_token)

        # Create an instance of the model
        model = cls(config=cfg, lang_embeddings=lang_embeddings, eos_indx=eos_indx, pad_index=pad_index)
        return model

    def feature_extractor_forward(self, src_tokens, encoder_padding_mask=None):
        # B x T x C x H x W
        if self.feature_extractor_type == "clip":
            B, T, _, _, _, = src_tokens.shape
            src_tokens = torch.flatten(src_tokens, start_dim=0, end_dim=1) # B x T x C x H x W -> (B x T) x C x H x W
            src_tokens = self.feature_extractor.get_image_features(pixel_values=src_tokens)
            src_tokens = torch.unflatten(src_tokens, 0, (B, T)) # (B x T) x C -> B x T x C
        return src_tokens, encoder_padding_mask
        
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
        if self.pad_index is not None and self.eos_indx is not None:
            padding_token = self.backbone.encoder.embed_tokens(torch.tensor([self.pad_index], dtype=torch.long, device=encoder_padding_mask.device))
            eos_token = self.backbone.encoder.embed_tokens(torch.tensor([self.eos_indx], dtype=torch.long, device=encoder_padding_mask.device))
            # Adjust <pad> tokens according the exped ones by the pretrained LM.
            bool_padding_mask = encoder_padding_mask == 0
            x[bool_padding_mask] = padding_token
        
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
            new_padding_column = padding_token.repeat(x.size(0), 1).unsqueeze(1)
            x = torch.cat([x, new_padding_column], dim=1)
            x[eos_inster_mask] = eos_token
        x = self.embed_scale * x
            
        return x, encoder_padding_mask

    def forward(self, input_ids=None, input_frames=None, src_langtoks=None, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None, ntokens=None):
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
        
        if self.feature_extractor is not None:
            input_frames, attention_mask = self.feature_extractor_forward(input_frames, attention_mask)
            
        if self.vl_mapper is not None:
            input_frames = self.vl_mapper(input_frames)
            
        input_frames, attention_mask =  self.forward_special_tokens(input_frames, attention_mask, src_langtoks)
        
        if self.config.backbone_name == "m2m100":
            outputs = self.backbone(
                inputs_embeds=input_frames, # inputs_embeds expected to have shape of (batch_size, sequence_length, hidden_size)
                attention_mask=attention_mask, # attention_mask expected to have shape of (batch_size, sequence_length)
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

        elif self.config.backbone_name == "bert":
            ## TODO: Not implemented yet (The below code is just a placeholder code)
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            raise ValueError(f"Unknown backbone name: {self.config.backbone_name}")

        # Placeholder Code created just as an example.
        lm_logits = self.lm_head(outputs[0])
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_index)
            loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return (loss, outputs) if loss is not None else outputs
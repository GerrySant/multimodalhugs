#### MultimodalEmbeddingSequenceBuilder
import math
import torch
import torch.nn as nn
from typing import Optional
from multimodalhugs.modules import CustomEmbedding
from multimodalhugs.modules.utils import merge_modalities


class SpecialTokensEmbeddings(nn.Module):
    def __init__(
        self, 
        old_vocab_size: int,
        new_vocab_size: int,
        embed_dim: int,
        pad_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
    ):
        super(SpecialTokensEmbeddings, self).__init__()
        self.special_tokens_embeddings = CustomEmbedding(
            used_size=old_vocab_size, 
            num_new_token=new_vocab_size, 
            emb_dim=embed_dim
        )
        self.pad_idx = pad_idx if pad_idx is not None else 1
        self.eos_idx = eos_idx if eos_idx is not None else 2
    
    @classmethod
    def build_module(
        cls, 
        old_vocab_size: int,
        new_vocab_size: int,
        embed_dim: int,
        pad_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
        old_embs_weight = None
    ): 
        module = cls(
            old_vocab_size=old_vocab_size, 
            new_vocab_size=new_vocab_size, 
            embed_dim=embed_dim,
            pad_idx=pad_idx,
            eos_idx=eos_idx,
        )
        if old_embs_weight is not None:
            custom_embeddings = CustomEmbedding.build_module(
                old_embs_weight, 
                backbone_used_vocab_size=old_vocab_size, 
                num_new_token=new_vocab_size, 
                emb_dim=embed_dim
            )
            module.special_tokens_embeddings = custom_embeddings
        return module
        
    def forward(self, x, encoder_padding_mask, src_prompt, source_prompt_length_padding_mask):
        return merge_modalities(
            x=x, 
            padding_mask=encoder_padding_mask, 
            prompt=src_prompt, 
            prompt_length_padding_mask=source_prompt_length_padding_mask,
            embeddings_module=self.special_tokens_embeddings, 
            pad_idx=self.pad_idx, 
            eos_idx=self.eos_idx, 
        )
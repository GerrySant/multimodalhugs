#### MultimodalEmbeddingSequenceBuilder
import math
import torch
import torch.nn as nn
from typing import Optional


class SpecialTokensEmbeddings(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embed_dim: int,
        scale_embeddings: bool = True,
        pad_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
    ):
        super(SpecialTokensEmbeddings, self).__init__()
        self.special_tokens_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.embed_scale = math.sqrt(embed_dim) if scale_embeddings else 1.0
        self.pad_idx = pad_idx if pad_idx is not None else 1
        self.eos_idx = eos_idx if eos_idx is not None else 2
        
    def forward(self, x, encoder_padding_mask, src_prompt, source_prompt_length_padding_mask):
        """
        It adds and/or corrects the special tokens from the input secuence:
            # '<prompt_token_1>', ..., '<prompt_token_N>', ...,  '</s>', '<pad>', '<pad>'
        
        INPUTS:
            - x: B x N_tokens x Embed_dim
            - encoder_padding_mask: B x N_tokens <- 0 indicates padding elements
            - src_prompt: B x N_tokens_prompt
            - source_prompt_length_padding_mask: B x N_tokens_prompt  
        """
        # Append <src_prompt>:
        if src_prompt is not None:
            src_prompt = self.special_tokens_embeddings(src_prompt)
            
            x = torch.cat((src_prompt, x), dim=1)

            # Correct Padding Mask
            if source_prompt_length_padding_mask is None:
                source_prompt_length_padding_mask = torch.full((encoder_padding_mask.size(0), x.size(1) - encoder_padding_mask.size(1)), 1, dtype=encoder_padding_mask.dtype, device=encoder_padding_mask.device) # torch.Size([B, 1])
            encoder_padding_mask = torch.cat([source_prompt_length_padding_mask, encoder_padding_mask], dim=1) # torch.Size([B, N_tokens_prompt]) + torch.Size([B, N_tokens]) = torch.Size([B, N_tokens + N_tokens_prompt])

        
        # Adjust <pad> tokens and add <eos> token to every secuence in the batch:
        if self.pad_idx is not None and self.eos_idx is not None:
            
            pad_token = self.special_tokens_embeddings(torch.tensor([self.pad_idx]).to(encoder_padding_mask.device))
            eos_token = self.special_tokens_embeddings(torch.tensor([self.eos_idx]).to(encoder_padding_mask.device))

            # Adjust <pad> tokens according the exped ones by the pretrained LM.
            bool_padding_mask = encoder_padding_mask == 0
            x[bool_padding_mask] = pad_token

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
            new_padding_column = pad_token.repeat(x.size(0), 1).unsqueeze(1)
            x = torch.cat([x, new_padding_column], dim=1)
            x[eos_inster_mask] = eos_token
        x = self.embed_scale * x
        return x, encoder_padding_mask
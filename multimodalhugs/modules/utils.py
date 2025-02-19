# Standard libraries
import logging

# Third-party libraries
import torch
import torch.nn as nn

# Other
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch.nn as nn

def extend_all_embeddings_and_lm_head(backbone, num_new_tokens, verbose=False):
    """
    Recursively search the backbone for any embedding layers (named 'embed_tokens' or 'shared')
    and nn.Linear layers named 'lm_head', and extend their number of rows by adding num_new_tokens.
    For embeddings, rows correspond to token indices; for lm_head, rows correspond to output features.
    Pretrained weights for the existing tokens/features are maintained and new rows are initialized 
    using a normal distribution. If an embedding layer is of a custom type (e.g. M2M100ScaledWordEmbedding),
    it is re-instantiated using its class and parameters.
    
    Additionally, for lm_head the function determines whether bias is used and extends bias accordingly.
    
    Args:
        backbone: The backbone module (e.g., model.backbone or model.backbone.model) to search.
        num_new_tokens (int): The number of new vocabulary tokens to add.
        verbose (bool): If True, prints details about each modified module.
        
    Returns:
        backbone: The updated backbone module with extended embedding layers and lm_head.
        new_vocab_size (int): The updated total number of embeddings (assumed consistent across updated layers).
    """
    new_vocab_size = None  # To store the new total number of embeddings

    def recursive_extend(module, prefix=""):
        nonlocal new_vocab_size
        for name, child in module.named_children():
            current_prefix = f"{prefix}.{name}" if prefix else name
            # If the module is an embedding layer named "embed_tokens" or "shared"
            if isinstance(child, nn.Embedding) and name in ("embed_tokens", "shared"):
                old_num_embeddings, embed_dim = child.weight.size()
                updated_vocab_size = old_num_embeddings + num_new_tokens
                # If the embedding layer is of a custom type (like M2M100ScaledWordEmbedding), re-instantiate using its class.
                if child.__class__.__name__ == "M2M100ScaledWordEmbedding":
                    new_embed = child.__class__(updated_vocab_size, embed_dim, child.padding_idx, embed_scale=child.embed_scale)
                else:
                    new_embed = nn.Embedding(updated_vocab_size, embed_dim)
                # Copy existing weights.
                new_embed.weight.data[:old_num_embeddings] = child.weight.data.clone()
                std_val = child.weight.std().item()  # Convert std tensor to float
                nn.init.normal_(new_embed.weight.data[old_num_embeddings:], mean=0.0, std=std_val)
                setattr(module, name, new_embed)
                if verbose:
                    print(f"Modified {current_prefix}: extended from {old_num_embeddings} to {updated_vocab_size} embeddings.")
                if new_vocab_size is None:
                    new_vocab_size = updated_vocab_size
                elif new_vocab_size != updated_vocab_size:
                    new_vocab_size = max(new_vocab_size, updated_vocab_size)
            # If the module is a linear layer named "lm_head"
            elif isinstance(child, nn.Linear) and name == "lm_head":
                old_vocab_size = child.out_features
                hidden_dim = child.in_features
                updated_vocab_size = old_vocab_size + num_new_tokens
                # Check if lm_head has a bias.
                bias_flag = child.bias is not None
                if bias_flag:
                    new_lm_head = nn.Linear(hidden_dim, updated_vocab_size, bias=True)
                else:
                    new_lm_head = nn.Linear(hidden_dim, updated_vocab_size, bias=False)
                # Copy old weight values.
                new_lm_head.weight.data[:old_vocab_size] = child.weight.data.clone()
                std_val = child.weight.std().item()
                nn.init.normal_(new_lm_head.weight.data[old_vocab_size:], mean=0.0, std=std_val)
                # If bias exists, copy old bias and initialize the new bias entries.
                if bias_flag:
                    new_lm_head.bias.data[:old_vocab_size] = child.bias.data.clone()
                    nn.init.zeros_(new_lm_head.bias.data[old_vocab_size:])
                setattr(module, name, new_lm_head)
                if verbose:
                    print(f"Modified {current_prefix}: extended lm_head from {old_vocab_size} to {updated_vocab_size} outputs.")
            else:
                # Recursively search this child module.
                recursive_extend(child, current_prefix)
    
    recursive_extend(backbone)
    return backbone, new_vocab_size

def set_module_parameters(module, freeze=True, verbose=True):
    """
    Set the parameters of the provided module to either freeze or unfreeze them.

    Args:
        module (torch.nn.Module): The module whose parameters are to be modified.
        freeze (bool): If True, freeze the module's parameters; if False, unfreeze them.
                       Defaults to True.
    """
    if module is None:
        return
    
    for param in module.parameters():
        param.requires_grad = not freeze

    action = "frozen" if freeze else "unfrozen"
    if verbose:
        logger.info(f" The parameters of the module {module.__class__.__name__} have been {action}.")


def merge_modalities(x, padding_mask, prompt, prompt_length_padding_mask,
                                       embeddings_module, pad_idx, eos_idx):
    """
    Adjusts the input sequence by adding and/or correcting special tokens:
      - Prepends prompt tokens if provided.
      - Adjusts the padding mask accordingly.
      - Replaces padding positions with the pad token embedding.
      - Inserts an end-of-sequence (EOS) token in each sequence.

    Parameters:
        x (Tensor): Input tensor of shape [B, N_tokens, Embed_dim].
        padding_mask (Tensor): Padding mask of shape [B, N_tokens] (0 indicates padding).
        prompt (Tensor or None): Prompt tensor of shape [B, N_tokens_prompt] or None.
        prompt_length_padding_mask (Tensor or None): Padding mask for prompt tokens of shape [B, N_tokens_prompt] or None.
        embeddings_module (Callable): Function or module to obtain embeddings for special tokens.
        pad_idx (int or None): Index of the <pad> token.
        eos_idx (int or None): Index of the </s> (EOS) token.

    Returns:
        Tuple[Tensor, Tensor]: The modified input tensor and the adjusted padding mask.
    """
    # Prepend prompt tokens if provided
    if prompt is not None:
        prompt = embeddings_module(prompt)
        x = torch.cat((prompt, x), dim=1)

        # Adjust the padding mask for the prompt tokens
        if prompt_length_padding_mask is None:
            prompt_length_padding_mask = torch.full(
                (padding_mask.size(0), x.size(1) - padding_mask.size(1)),
                1, dtype=padding_mask.dtype, device=padding_mask.device)
        padding_mask = torch.cat([prompt_length_padding_mask, padding_mask], dim=1)

    # Replace <pad> tokens and add an EOS token to each sequence in the batch
    if pad_idx is not None and eos_idx is not None:
        pad_token = embeddings_module(torch.tensor([pad_idx]).to(padding_mask.device))
        eos_token = embeddings_module(torch.tensor([eos_idx]).to(padding_mask.device))

        # Replace padding positions with the pad token embedding
        bool_padding_mask = padding_mask == 0
        x[bool_padding_mask] = pad_token

        # Adjust the padding mask to reflect the addition of the EOS token
        new_mask_entry = torch.full(
            (padding_mask.size(0), 1),
            1, dtype=padding_mask.dtype, device=padding_mask.device)
        padding_mask = torch.cat([new_mask_entry, padding_mask], dim=1)

        # Create a mask indicating where to insert the EOS token in each sequence
        eos_insert_mask = torch.zeros_like(padding_mask)
        last_indices = padding_mask.size(1) - torch.argmax(padding_mask.flip(dims=[1]), dim=1) - 1
        rows = torch.arange(padding_mask.size(0))
        eos_insert_mask[rows, last_indices] = 1
        eos_insert_mask = eos_insert_mask != 0

        # Append an extra padding column to prepare for EOS token insertion, then insert the EOS token
        new_padding_column = pad_token.repeat(x.size(0), 1).unsqueeze(1)
        x = torch.cat([x, new_padding_column], dim=1)
        x[eos_insert_mask] = eos_token

    return x, padding_mask

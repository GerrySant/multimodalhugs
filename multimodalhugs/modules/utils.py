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

def extend_all_embeddings_and_lm_head(backbone, num_new_tokens, pad_to_multiple_of=None, verbose=False):
    """
    Extends the backbone's input embeddings and lm_head (if not tied) to incorporate additional tokens,
    using the backbone's built-in resize methods.

    This function:
      1. Determines the new vocabulary size as:
             new_vocab_size = current_vocab_size + num_new_tokens
      2. Calls backbone.resize_token_embeddings(new_num_tokens=new_vocab_size, ...)
         to update input embeddings (and backbone.shared if present).
      3. If the output embeddings (lm_head) exist and are not tied, it resizes them using the backbone's 
         _get_resized_embeddings or _get_resized_lm_head method.
      4. Prints verbose messages if requested.

    Args:
        backbone: The backbone module. It must have methods like get_input_embeddings(),
                  set_input_embeddings(), resize_token_embeddings(), get_output_embeddings(), etc.
        num_new_tokens (int): Number of new tokens to add.
        pad_to_multiple_of (int, optional): Passed to resize_token_embeddings to pad the vocab size.
        verbose (bool): If True, prints detailed messages.

    Returns:
        backbone: The updated backbone module.
        new_vocab_size (int): The new total vocabulary size.
    """
    # Get current input embeddings.
    input_embeddings = backbone.get_input_embeddings()
    if input_embeddings is None:
        if verbose:
            print("No input embeddings found in the backbone.")
        return backbone, None

    # Determine current vocabulary size.
    current_vocab_size = input_embeddings.weight.size(0)
    new_vocab_size = current_vocab_size + num_new_tokens
    if verbose:
        print(f"Current vocabulary size: {current_vocab_size}. Adding {num_new_tokens} tokens to get {new_vocab_size}.")

    # Resize the token embeddings (and shared embeddings, if present).
    backbone.resize_token_embeddings(new_num_tokens=new_vocab_size, pad_to_multiple_of=pad_to_multiple_of)
    if verbose:
        print(f"Resized token embeddings to {new_vocab_size}.")

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

def merge_modalities_mask_correction(padding_mask, prompt, prompt_length_padding_mask, embeddings_module, pad_idx, eos_idx):
    """
    Adjusts the padding mask for generation when x is None.
    
    This function simulates the mask correction that would normally be applied
    when merging modalities. It takes into account:
      - Prepending prompt tokens (using prompt_length_padding_mask, or ones if not provided).
      - Adding a special token column (for EOS, as done in the training merge).
    
    Parameters:
        padding_mask (Tensor): Original padding mask of shape [B, N_tokens] (0 indicates padding).
        prompt (Tensor or None): Prompt tensor of shape [B, N_tokens_prompt] or None.
        prompt_length_padding_mask (Tensor or None): Mask for prompt tokens of shape [B, N_tokens_prompt] or None.
        embeddings_module (Callable): Unused here, kept for interface consistency.
        pad_idx (int or None): Index of the <pad> token.
        eos_idx (int or None): Index of the </s> (EOS) token.
    
    Returns:
        Tensor: The corrected padding mask reflecting the addition of prompt and special tokens.
    """
    # If a prompt is provided, prepend its corresponding mask.
    if prompt is not None:
        if prompt_length_padding_mask is None:
            prompt_length = prompt.size(1)
            prompt_length_padding_mask = torch.ones(
                (padding_mask.size(0), prompt_length),
                dtype=padding_mask.dtype,
                device=padding_mask.device
            )
        padding_mask = torch.cat([prompt_length_padding_mask, padding_mask], dim=1)
    
    # If special token indices are provided, prepend a column to account for the special token (e.g. EOS)
    if pad_idx is not None and eos_idx is not None:
        new_mask_entry = torch.ones(
            (padding_mask.size(0), 1),
            dtype=padding_mask.dtype,
            device=padding_mask.device
        )
        padding_mask = torch.cat([new_mask_entry, padding_mask], dim=1)
    
    return padding_mask

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

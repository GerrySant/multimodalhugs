import os
import copy
import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

def load_tokenizer_from_vocab_file(vocab_file, special_tokens_dict=None, output_dir=None):
    vocab = {}
    special_tokens = {
        '<s>': 0,   # bos_token
        '</s>': 2,  # eos_token, sep_token
        '<unk>': 3, # unk_token
        '<pad>': 1  # pad_token
    } if special_tokens_dict is None else special_tokens_dict

    with open(vocab_file, 'r') as f:
        for line in f:
            token, _ = line.strip().split()
            vocab[token] = len(vocab) + len(special_tokens)

    combined_vocab = dict(sorted({**special_tokens, **vocab}.items(), key=lambda item: item[1]))

    # Save vocab to JSON file
    vocab_json_path = vocab_file.replace('.txt', '.json')
    if output_dir is not None:
        vocab_json_path = os.path.join(output_dir, os.path.basename(vocab_json_path))
    with open(vocab_json_path, 'w') as f:
        json.dump(combined_vocab, f, indent=4)

    # Initialize a tokenizer
    tokenizer = Tokenizer(WordLevel(vocab=combined_vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Save the tokenizer
    tokenizer_path = vocab_json_path.replace('.json', '_tokenizer.json')
    tokenizer.save(tokenizer_path)

    # Load the tokenizer with HuggingFace transformers
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, additional_special_tokens=list(vocab.keys()))
    fast_tokenizer.add_special_tokens({
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'sep_token': '</s>',
        'pad_token': '<pad>'
    })

    # Save the HuggingFace tokenizer
    fast_tokenizer.save_pretrained(tokenizer_path.replace('_tokenizer.json', '_tokenizer'))
    return fast_tokenizer


def add_new_special_tokens_from_vocab_file(tokenizer, vocab_file, output_dir=None):
    """
    Adds new special tokens to the tokenizer from either:
      - a vocabulary file (one token per line, optional whitespace/comments), or
      - a comma-separated string of tokens.
    Skips any token already present in the tokenizer.
    
    Args:
        tokenizer: The tokenizer instance.
        vocab_file: str, either a path to a file or a comma-separated string.
        output_dir: Optional directory to save the updated tokenizer.

    Returns:
        tokenizer: The updated tokenizer.
        added_tokens: List of tokens added.
    """
    # Determine if vocab_file is a file path or a comma-separated string
    if not isinstance(vocab_file, str) or not vocab_file.strip():
        raise ValueError("vocab_file must be a non-empty string (file path or comma-separated list) or None if no new_vocabulary is needed.")

    if os.path.isfile(vocab_file):
        with open(vocab_file, 'r') as f:
            raw_tokens = [line.strip().split()[0] for line in f if line.strip()]
    else:
        # assume it's a comma-separated string
        raw_tokens = [tok.strip() for tok in vocab_file.split(',') if tok.strip()]

    added_tokens = []
    skipped_tokens = []

    existing_vocab = tokenizer.get_vocab()
    for token in raw_tokens:
        if token not in existing_vocab:
            added_tokens.append(token)
        else:
            skipped_tokens.append(token)

    if added_tokens:
        tokenizer.add_special_tokens(
            {'additional_special_tokens': added_tokens},
            replace_additional_special_tokens=False
        )
        print(f"Added tokens: {added_tokens}")
    else:
        print("No new tokens to add.")

    if skipped_tokens:
        print(f"Skipped tokens (already present): {skipped_tokens}")

    if output_dir is not None:
        tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tokenizer_output_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_output_dir)
        print(f"Tokenizer saved to {tokenizer_output_dir}")

    return tokenizer, added_tokens


def extend_tokenizer(tokenizer_path, new_vocabulary, training_output_dir=None, model_name=None):
    """
    Load a pretrained tokenizer and optionally extend it with tokens from a vocabulary file.
    The extended tokenizer is **saved to disk only if all of the following are true**:
    1) `new_vocabulary` is not None,
    2) `training_output_dir` is not None, and
    3) `model_name` is not None.
    In that case, the save path passed downstream is `os.path.join(training_output_dir, model_name)`.
    If any of these are missing, the tokenizer is updated in-memory only and not saved.

    Args:
        tokenizer_path (str): Hugging Face identifier or local path to the pretrained tokenizer.
        new_vocabulary (str | None): Path to a text file with tokens to add (e.g., one per line).
        training_output_dir (str | None): Base output directory for saving the extended tokenizer.
        model_name (str | None): Subdirectory name under `training_output_dir` used for saving.

    Returns:
        AutoTokenizer: The updated tokenizer.
        list[str]: Newly added special tokens.
    """
    # Load the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    new_vocab_tokens = []
    
    # If new_vocabulary is provided, add the tokens
    if new_vocabulary is not None:
        output_dir = os.path.join(training_output_dir, model_name) if training_output_dir is not None and model_name is not None else None
        tokenizer, new_special_tokens = add_new_special_tokens_from_vocab_file(
            tokenizer=copy.deepcopy(tokenizer),
            vocab_file=new_vocabulary,
            output_dir=output_dir,
        )
        new_vocab_tokens.extend(new_special_tokens)
    
    return tokenizer, new_vocab_tokens
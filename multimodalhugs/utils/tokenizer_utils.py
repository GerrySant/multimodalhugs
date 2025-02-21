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
    Reads a vocabulary file and adds new special tokens to the tokenizer,
    skipping any token that the tokenizer already has.
    """
    added_tokens = []
    skipped_tokens = []

    with open(vocab_file, 'r') as f:
        for line in f:
            token = line.strip().split()[0]
            if token not in tokenizer.get_vocab():
                added_tokens.append(token)
            else:
                skipped_tokens.append(token)

    print(f"The following tokens have been added to the tokenizer: {added_tokens}")
    print(f"The following tokens have not been added because the tokenizer already has them: {skipped_tokens}")

    if added_tokens:
        tokenizer.add_special_tokens(
            special_tokens_dict={'additional_special_tokens': added_tokens},
            replace_additional_special_tokens=False
        )

    if output_dir is not None:
        tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
        # Optionally, save the updated tokenizer to tokenizer_output_dir

    return tokenizer, added_tokens


def extend_tokenizer(dataset_config, training_output_dir=None, model_name=None):
    """
    Loads a pretrained tokenizer based on dataset_config.text_tokenizer_path and extends it
    with the tokens specified in dataset_config.new_vocabulary.
    
    Args:
        dataset_config: A configuration object that must contain:
            - text_tokenizer_path: (str) Identifier or path for the pretrained tokenizer.
            - new_vocabulary: (str) Path to the file with new tokens.
        training_output_dir: (str) Base output directory (e.g., config.training.output_dir).
        model_name: (str) Name of the model (used to determine the output directory for the vocab update).
    
    Returns:
        tokenizer: The updated tokenizer (an instance of AutoTokenizer).
        new_vocab_tokens: A list of new special tokens that were added.
    """
    # Load the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(dataset_config.text_tokenizer_path)
    new_vocab_tokens = []
    
    # If new_vocabulary is provided, add the tokens
    if getattr(dataset_config, "new_vocabulary", None):
        output_dir = os.path.join(training_output_dir, model_name) if training_output_dir is not None and model_name is not None else None
        tokenizer, new_special_tokens = add_new_special_tokens_from_vocab_file(
            tokenizer=copy.deepcopy(tokenizer),
            vocab_file=dataset_config.new_vocabulary,
            output_dir=output_dir,
        )
        new_vocab_tokens.extend(new_special_tokens)
    
    return tokenizer, new_vocab_tokens
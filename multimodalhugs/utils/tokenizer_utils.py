import os
import copy
from transformers import AutoTokenizer
from multimodalhugs.data import add_new_special_tokens_from_vocab_file

def extend_tokenizer(dataset_config, training_output_dir, model_name):
    """
    Loads a pretrained tokenizer based on dataset_config.text_tokenizer_path and extends it with any new vocabulary tokens
    specified in the dataset configuration. This function is meant to be used in training setup scripts for different modalities.
    
    Args:
        dataset_config: A configuration object (or namespace/dict) that must contain:
            - text_tokenizer_path: (str) Identifier or path for the pretrained tokenizer.
            - tokenizer_src_langs_path: (optional str) Path to a file with extra tokens for source languages.
            - new_task_tokens_dictionary_path: (optional str) Path to a file with extra tokens for new tasks.
        training_output_dir: (str) Base output directory (e.g. config.training.output_dir).
        model_name: (str) Name of the model (used to determine the output directory for the last vocab update).
    
    Returns:
        tokenizer: The updated tokenizer (an instance of AutoTokenizer).
        new_vocab_tokens: A list of the new special tokens that were added.
    """
    # Load the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(dataset_config.text_tokenizer_path)
    new_vocab_tokens = []
    vocab_files = []

    # If provided, add vocabulary files from the configuration
    if getattr(dataset_config, "tokenizer_src_langs_path", None):
        vocab_files.append(dataset_config.tokenizer_src_langs_path)
    if getattr(dataset_config, "new_task_tokens_dictionary_path", None):
        vocab_files.append(dataset_config.new_task_tokens_dictionary_path)

    # For each vocab file, update the tokenizer
    for i, vocab_file in enumerate(vocab_files):
        output_dir = None
        # If this is the last vocab file, set the output directory accordingly.
        if i == len(vocab_files) - 1:
            output_dir = os.path.join(training_output_dir, model_name)
        # Use a copy of the tokenizer so that modifications don’t accumulate in an unexpected way.
        tokenizer, new_special_tokens = add_new_special_tokens_from_vocab_file(
            tokenizer=copy.deepcopy(tokenizer),
            vocab_file=vocab_file,
            output_dir=output_dir,
        )
        new_vocab_tokens.extend(new_special_tokens)

    return tokenizer, new_vocab_tokens

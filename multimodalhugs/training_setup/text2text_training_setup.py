import os
import copy
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from multimodalhugs.data.datasets.bilingual_image2text import (
    BilingualText2TextDataset,
)
from multimodalhugs.data.dataset_configs.multimodal_mt_data_config import MultimodalMTDataConfig
from multimodalhugs.processors import Text2TextTranslationProcessor
from multimodalhugs.utils.registry import get_model_class
from multimodalhugs.utils.utils import add_argument_to_the_config, reformat_yaml_file
from multimodalhugs.utils.tokenizer_utils import extend_tokenizer

from transformers import AutoTokenizer

def main(config_path):
    # Load config and initialize dataset
    config = OmegaConf.load(config_path)
    dataset_config = MultimodalMTDataConfig(config)
    dataset = BilingualText2TextDataset(config=dataset_config)

    # Download, prepare, and save dataset
    if getattr(dataset_config, 'dataset_dir', None) is not None and os.path.exists(dataset_config.dataset_dir):
        data_path = dataset_config.dataset_dir
    else:
        data_path = Path(config.training.output_dir) / config.training.run_name / "datasets" / dataset.name
        if not data_path.exists():
            # Download, prepare, and save dataset only if data_path doesn't exist
            dataset.download_and_prepare(data_path)
            dataset.as_dataset().save_to_disk(data_path)

    # Load the tokenizer (here, we use AutoTokenizer)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(dataset_config.text_tokenizer_path)
    tokenizer, new_vocab_tokens = extend_tokenizer(
        dataset_config, 
        training_output_dir=config.training.output_dir, 
        model_name=config.training.run_name
    )

    # Create the image-to-text processor
    input_processor = Text2TextTranslationProcessor(
        tokenizer=tokenizer,
    )

    # Save processor and define the processor path
    processor_path = os.path.join(config.training.output_dir, config.training.run_name, "text2text_translation_processor")
    input_processor.save_pretrained(save_directory=processor_path, push_to_hub=False)

    # --- Model creation becomes model-independent ---
    # Use a "type" field in config.model to select the model type (defaulting to "multimodal_embedder")
    model_type = config.model.get("type", "multimodal_embedder")
    if model_type is None:
        raise ValueError("model_type not found. Please specify a valid model type on the config.")
    model_class = get_model_class(model_type)

    # Convert the model section of the config to a dictionary and update with common arguments
    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    model_kwargs.update({
        "src_tokenizer": tokenizer,
        "tgt_tokenizer": pretrained_tokenizer,
        "config_path": config_path,
        "new_vocab_tokens": new_vocab_tokens,
    })

    # Build the model using the registry lookup and passing all configuration parameters as kwargs
    model = model_class.build_model(**model_kwargs)
    # Save the model and print the paths for later use
    model_path = os.path.join(config.training.output_dir, config.training.run_name, "trained_model")
    model.save_pretrained(model_path)

    add_argument_to_the_config(config_path, "processor", "processor_name_or_path", str(processor_path))
    add_argument_to_the_config(config_path, "data", "dataset_dir", str(data_path))
    add_argument_to_the_config(config_path, "model", "model_name_or_path", str(model_path))
    reformat_yaml_file(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training setup for multimodal models")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_path)

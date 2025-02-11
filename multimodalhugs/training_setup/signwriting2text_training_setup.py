# signwriting2text_training_setup.py
#
# Usage: python signwriting2text_training_setup.py --config_path path_to_your_config.yaml

import os
import copy
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from multimodalhugs.data import SignWritingDataset, MultimodalMTDataConfig, add_new_special_tokens_from_vocab_file
from multimodalhugs.processors import SignwritingProcessor
from transformers import AutoTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

# Import the model registry helper (ensure this module exists)
from multimodalhugs.models.registry import get_model_class
from multimodalhugs.utils.tokenizer_utils import extend_tokenizer

def main(config_path):
    # Load config and initialize dataset
    config = OmegaConf.load(config_path)
    dataset_config = MultimodalMTDataConfig(config)
    dataset = SignWritingDataset(config=dataset_config)

    # Download, prepare, and save dataset
    data_path = Path(config.training.output_dir) / config.model.name / "datasets" / dataset.name
    dataset.download_and_prepare(data_path)
    dataset.as_dataset().save_to_disk(data_path)

    # Load the tokenizer (here, we use AutoTokenizer)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(dataset_config.text_tokenizer_path)
    tokenizer, new_vocab_tokens = extend_tokenizer(
        dataset_config, 
        training_output_dir=config.training.output_dir, 
        model_name=config.model.name
    )

    # Create frame preprocessor for image inputs
    frame_preprocessor = CLIPImageProcessor(
        do_resize=dataset_config.preprocess.do_resize,
        size=dataset_config.preprocess.width,
        do_center_crop=dataset_config.preprocess.do_center_crop,
        do_rescale=dataset_config.preprocess.do_rescale,
        do_normalize=dataset_config.preprocess.do_normalize,
        image_mean=dataset_config.preprocess.dataset_mean,
        image_std=dataset_config.preprocess.dataset_std,
    )

    # Create the processor
    input_processor = SignwritingProcessor(
        width=dataset_config.preprocess.width,
        height=dataset_config.preprocess.height,
        channels=dataset_config.preprocess.channels,
        invert_frame=dataset_config.preprocess.invert_frame,
        dataset_mean=dataset_config.preprocess.dataset_mean,
        dataset_std=dataset_config.preprocess.dataset_std,
        frame_preprocessor=frame_preprocessor,
        tokenizer=tokenizer,
    )

    # Save processor and set PROCESSOR_PATH environment variable
    processor_path = os.path.join(config.training.output_dir, config.model.name, "signwriting_processor")
    input_processor.save_pretrained(save_directory=processor_path, push_to_hub=False)

    # --- Model creation becomes model independent ---
    # Use the "type" field in config.model to select the appropriate model.
    # If not provided, default to "multimodal_embedder".
    model_type = config.model.get("type", "multimodal_embedder")
    model_class = get_model_class(model_type)

    # Convert the model section of the config to a dictionary.
    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    # Add common arguments required by build_model() to the keyword arguments.
    model_kwargs.update({
        "src_tokenizer": tokenizer,
        "tgt_tokenizer": pretrained_tokenizer,
        "config_path": config_path,
        "new_vocab_tokens": new_vocab_tokens,
    })
    # Build the model using the model class's build_model method.
    model = model_class.build_model(**model_kwargs)

    # Save the model
    model_path = os.path.join(config.training.output_dir, config.model.name, "trained_model")
    model.save_pretrained(model_path)

    print(f"MODEL_PATH={model_path}")
    print(f"PROCESSOR_PATH={processor_path}")
    print(f"DATA_PATH={data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training setup for multimodal models")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_path)

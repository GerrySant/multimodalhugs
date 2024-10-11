### Usage: python training_setup.py --config_path path_to_your_config.yaml

import os
import copy
import torch
import argparse
from omegaconf import OmegaConf
from pathlib import Path

from multimodalhugs.data import SignWritingDataset, MultimodalMTDataConfig, add_new_special_tokens_from_vocab_file
from multimodalhugs.processors import SignwritingProcessor
from multimodalhugs.models import MultiModalEmbedderModel

from transformers import M2M100Tokenizer, AutoTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

def main(config_path):
    # Load config and initialize dataset
    config = OmegaConf.load(config_path)
    dataset_config = MultimodalMTDataConfig(config)
    dataset = SignWritingDataset(config=dataset_config)

    # Download, prepare, and save dataset
    data_path = Path(config.training.output_dir) / config.model.name / "datasets" / dataset.name
    dataset.download_and_prepare(data_path)
    dataset.as_dataset().save_to_disk(data_path)

    frame_preprocessor = CLIPImageProcessor(
                do_resize=dataset_config.preprocess.do_resize,
                size=dataset_config.preprocess.width,
                do_center_crop=dataset_config.preprocess.do_center_crop,
                do_rescale=dataset_config.preprocess.do_rescale,
                do_normalize=dataset_config.preprocess.do_normalize,
                image_mean=dataset_config.preprocess.dataset_mean,
                image_std=dataset_config.preprocess.dataset_std,
            )

    m2m_tokenizer = AutoTokenizer.from_pretrained(dataset_config.text_tokenizer_path)
    tokenizer = add_new_special_tokens_from_vocab_file(
        tokenizer=copy.deepcopy(m2m_tokenizer), 
        vocab_file=dataset_config.src_lang_tokenizer_path,
        output_dir=config.training.output_dir + "/" + config.model.name,
    )

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
    processor_path = config.training.output_dir + f"/{config.model.name}" + f"/signwriting_processor"
    input_processor.save_pretrained(save_directory=processor_path, push_to_hub=False)

    # Build and save the model, then set MODEL_PATH environment variable
    model = MultiModalEmbedderModel.build_model(
        cfg=config.model, 
        src_tokenizer=tokenizer, 
        tgt_tokenizer=m2m_tokenizer
    )

    model_path = f"{config.training.output_dir}/{config.model.name}/trained_model"
    model.save_pretrained(model_path)

    print(f"MODEL_PATH={model_path}")
    print(f"PROCESSOR_PATH={processor_path}")
    print(f"DATA_PATH={data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for multimodal models")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    main(args.config_path)
### Usage: python training_setup.py --config_path path_to_your_config.yaml

import os
import copy
import torch
import argparse
from omegaconf import OmegaConf
from pathlib import Path

from multimodalhugs.data import How2SignDataset, SignLanguageMTDataConfig, add_new_special_tokens_from_vocab_file
from multimodalhugs.processors import Pose2TextTranslationProcessor
from multimodalhugs.models import MultiModalEmbedderModel

from transformers import AutoTokenizer

def main(config_path):
    # Load config and initialize dataset
    config = OmegaConf.load(config_path)
    dataset_config = SignLanguageMTDataConfig(config)
    dataset = How2SignDataset(config=dataset_config)

    # Download, prepare, and save dataset
    data_path = Path(config.training.output_dir) / config.model.name / "datasets" / dataset.name
    dataset.download_and_prepare(data_path)
    dataset.as_dataset().save_to_disk(data_path)

    m2m_tokenizer = AutoTokenizer.from_pretrained(dataset_config.text_tokenizer_path)
    tokenizer = add_new_special_tokens_from_vocab_file(
        tokenizer=copy.deepcopy(m2m_tokenizer), 
        vocab_file=dataset_config.src_lang_tokenizer_path,
        output_dir=config.training.output_dir + "/" + config.model.name,
    )

    input_processor = Pose2TextTranslationProcessor(
            tokenizer=tokenizer,
            reduce_holistic_poses=True,
            target_lang_on_source=True,
            task_prefixes=[],
    )

    # Save processor and set PROCESSOR_PATH environment variable
    processor_path = config.training.output_dir + f"/{config.model.name}" + f"/pose2text_translation_processor"
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
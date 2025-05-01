### Usage: python training_setup.py --config_path path_to_your_config.yaml

import os
import copy
import torch
import argparse
from omegaconf import OmegaConf
from pathlib import Path

from multimodalhugs.data.datasets.video2text import Video2TextDataset, Video2TextDataConfig
from multimodalhugs.processors import Video2TextTranslationProcessor
from multimodalhugs.utils.registry import get_model_class
from multimodalhugs.utils.utils import add_argument_to_the_config, reformat_yaml_file
from multimodalhugs.utils.tokenizer_utils import extend_tokenizer

from transformers import AutoTokenizer

def main(config_path):
    # Load config and initialize dataset
    config = OmegaConf.load(config_path)
    dataset_config = Video2TextDataConfig(config)
    dataset = Video2TextDataset(config=dataset_config)

    # Download, prepare, and save dataset
    if getattr(dataset_config, 'dataset_dir', None) is not None and os.path.exists(dataset_config.dataset_dir):
        data_path = dataset_config.dataset_dir
    else:
        data_path = Path(config.training.output_dir) / "datasets" / dataset.name
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

    # The preprocessor is created
    input_processor = Video2TextTranslationProcessor(
            tokenizer=tokenizer,
            normalize=dataset_config.normalize,
            resize=dataset_config.resize,
            join_chw=dataset_config.join_chw
    )

    # Save processor and set PROCESSOR_PATH environment variable
    processor_path = os.path.join(config.training.output_dir, "video2text_translation_processor")
    input_processor.save_pretrained(save_directory=processor_path, push_to_hub=False)

    # --- Model creation becomes model-independent ---
    # Use the "type" field in the configuration (defaulting if not provided)
    model_type = config.model.get("type", None)
    if model_type is None:
        raise ValueError("model_type not found. Please specify a valid model type on the config.")
    model_class = get_model_class(model_type)

    # Convert the model section of the config to a dictionary.
    # This will include any extra parameters specific to the model.
    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    
    # Update with common arguments required by build_model().
    model_kwargs.update({
        "src_tokenizer": tokenizer,
        "tgt_tokenizer": pretrained_tokenizer,
        "config_path": config_path,
        "new_vocab_tokens": new_vocab_tokens,
    })

    # Build the model. Each model class can decide which arguments to use.
    model = model_class.build_model(**model_kwargs)

    model_path = os.path.join(config.training.output_dir, config.training.run_name)
    model.save_pretrained(model_path)

    add_argument_to_the_config(config_path, "processor", "processor_name_or_path", str(processor_path))
    add_argument_to_the_config(config_path, "data", "dataset_dir", str(data_path))
    add_argument_to_the_config(config_path, "model", "model_name_or_path", str(model_path))
    reformat_yaml_file(config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for multimodal models")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    main(args.config_path)

batch_inside: {'signal': ['/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/EQWFrWeRVjQ_5-5-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/eRiOhdeskNE_14-8-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/4XlVMRXLydg_21-5-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/3HCjTYIijec_2-2-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/-f1_kdl050s_1-1-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/eY32ru3Nstc_3-8-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/EjzQn4ReeeI_7-5-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/DfnHNkTE7mE_19-5-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/ETOZLBScxWY_15-5-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/CzkLI34HFIg_19-5-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/1-xK5UtDSmE_9-2-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/ETOZLBScxWY_1-5-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/DfnHNkTE7mE_22-5-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/a4uz0W33REs_2-3-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/2yudAtTnZrg_5-1-rgb_front.mp4', '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/1-xK5UtDSmE_10-2-rgb_front.mp4'], 'signal_start': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'signal_end': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'encoder_prompt': ['__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__', '__asl__'], 'decoder_prompt': ['__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__', '__en__'], 'output': ["Now, what you want to do here - this is kind of important - you want to take this first leg or one leg and you want to put it facing the inside of your drum set or facing where you're going to be sitting, because when you hit it from that angle you don't want it to tip over.", "Alright now we want to hang this off the wreath so I'm going to need a smaller hole.", "Then begin the posture and the movement is what we're going to do, which is gathering chi from heaven and earth.", "This is Dr. Paul, author of Boomer Girls, a boomer woman's guide to men and dating and host of Ask Dr. Paul.", 'The number one loss for these birds, is flight.', "Pour that into your glass, and then we're going to add to that, three fourths of an ounce of coconut rum, any brand will do, so choose your favorite.", 'Standard cartoon hands, with the four fingers.', "It was actually lower that the Screen Actors Guild's standard, and we all got a low budget pay for it.", 'And this is how, some repair tips for setting up your tent.', "So, in a minute I'm going to show you how, I'm going to talk about it and show you how we're going to trim this off.", "Another one is drinking water, we're going to get into these in a little more detail.", "The first thing you're going to run into when repairing a tent, is you normally get small holes, rips in your tent, the mesh and other parts of the tent.", 'So I was in a scene with Charlize Theron in Monster.', "SIG HAUER: And I'm Sig Hauer, and we're professional practitioners of traditional Chinese medicine.", 'Let me see.', 'Another one would be eating a proper diet.']}
batch_outside: [{'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/EQWFrWeRVjQ_5-5-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "Now, what you want to do here - this is kind of important - you want to take this first leg or one leg and you want to put it facing the inside of your drum set or facing where you're going to be sitting, because when you hit it from that angle you don't want it to tip over."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/eRiOhdeskNE_14-8-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "Alright now we want to hang this off the wreath so I'm going to need a smaller hole."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/4XlVMRXLydg_21-5-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "Then begin the posture and the movement is what we're going to do, which is gathering chi from heaven and earth."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/3HCjTYIijec_2-2-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "This is Dr. Paul, author of Boomer Girls, a boomer woman's guide to men and dating and host of Ask Dr. Paul."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/-f1_kdl050s_1-1-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': 'The number one loss for these birds, is flight.'}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/eY32ru3Nstc_3-8-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "Pour that into your glass, and then we're going to add to that, three fourths of an ounce of coconut rum, any brand will do, so choose your favorite."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/EjzQn4ReeeI_7-5-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': 'Standard cartoon hands, with the four fingers.'}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/DfnHNkTE7mE_19-5-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "It was actually lower that the Screen Actors Guild's standard, and we all got a low budget pay for it."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/ETOZLBScxWY_15-5-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': 'And this is how, some repair tips for setting up your tent.'}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/CzkLI34HFIg_19-5-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "So, in a minute I'm going to show you how, I'm going to talk about it and show you how we're going to trim this off."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/1-xK5UtDSmE_9-2-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "Another one is drinking water, we're going to get into these in a little more detail."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/ETOZLBScxWY_1-5-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "The first thing you're going to run into when repairing a tent, is you normally get small holes, rips in your tent, the mesh and other parts of the tent."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/DfnHNkTE7mE_22-5-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': 'So I was in a scene with Charlize Theron in Monster.'}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/a4uz0W33REs_2-3-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': "SIG HAUER: And I'm Sig Hauer, and we're professional practitioners of traditional Chinese medicine."}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/2yudAtTnZrg_5-1-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': 'Let me see.'}, {'signal': '/home/gsantm/store/data/How2Sign/How2Sign/sentence_level/val/rgb_front/raw_videos/1-xK5UtDSmE_10-2-rgb_front.mp4', 'signal_start': 0, 'signal_end': 0, 'encoder_prompt': '__asl__', 'decoder_prompt': '__en__', 'output': 'Another one would be eating a proper diet.'}]
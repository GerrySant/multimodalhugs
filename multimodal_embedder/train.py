
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from omegaconf import OmegaConf

# Load custom modules (these need to be correctly defined in your project structure)
from multimodal_embedder.data.signwriting import SignWritingDataset
from multimodal_embedder.data.data_configs import MultimodalMTDataConfig
from multimodal_embedder.data.utils import _transform
from multimodal_embedder.models import MultiModalEmbedderModel

def load_config(config_path):
    """Load configuration from a YAML file."""
    return OmegaConf.load(config_path)

def initialize_dataset(cfg, split):
    """Initialize the SignWriting dataset."""
    preprocess_fn = _transform(
        cfg.preprocess.width, 
        cfg.preprocess.dataset_mean, 
        cfg.preprocess.dataset_std
    )
    dataset = SignWritingDataset(
        metafile_path=cfg.train_path if split == 'train' else cfg.val_path, 
        split='train', 
        config=cfg, 
        preprocess_fn=preprocess_fn
    )
    return dataset

def initialize_dataloader(dataset, batch_size, shuffle):
    """Set up DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)

def train_model(model, train_data_loader, val_data_loader, collate_fn, cfg):
    """Train the model with the given DataLoader."""
    training_args = TrainingArguments(
        **cfg.training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data_loader,
        eval_dataset=val_data_loader,
        data_collator=collate_fn
    )

    trainer.train()

def main():
    parser = argparse.ArgumentParser(description='Train a model with a specified configuration file.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    
    cfg = load_config(args.config_path)
    data_cfg = MultimodalMTDataConfig(cfg)

    train_dataset = initialize_dataset(data_cfg, 'train')
    val_dataset = initialize_dataset(data_cfg, 'val')

    # preprocess_fn = _transform(
    #     data_cfg.preprocess.width, 
    #     data_cfg.preprocess.dataset_mean, 
    #     data_cfg.preprocess.dataset_std
    # )

    # # Initialize dataset
    # train_dataset = SignWritingDataset(metafile_path=data_cfg.train_path, split='train', config=data_cfg, preprocess_fn=preprocess_fn)
    # val_dataset = SignWritingDataset(metafile_path=data_cfg.val_path, split='train', config=data_cfg, preprocess_fn=preprocess_fn)

    model = MultiModalEmbedderModel.build_model(cfg.model, train_dataset)

    train_model(model, train_dataset, val_dataset, train_dataset.collate_fn, cfg)

    model.save_pretrained(cfg.training.output_dir + "/trained_model")
    train_dataset.src_tokenizer.save_pretrained(cfg.training.output_dir + "/trained_src_tokenizer")
    val_dataset.tgt_tokenizer.save_pretrained(cfg.training.output_dir + "/trained_tgt_tokenizer")

if __name__ == "__main__":
    main()

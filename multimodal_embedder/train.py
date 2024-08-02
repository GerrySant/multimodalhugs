import torch
import torch.nn as nn
import argparse
import logging
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from omegaconf import OmegaConf

# Load custom modules (these need to be correctly defined in your project structure)
from multimodal_embedder.data.signwriting import SignWritingDataset
from multimodal_embedder.data.data_configs import MultimodalMTDataConfig
from multimodal_embedder.data.utils import _transform
from multimodal_embedder.models import MultiModalEmbedderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_module_details(model):
    # Define column widths
    col_widths = [20, 17, 12]
    
    # Start building the output string
    output = []
    
    # Header and line setup
    header = "Model Summary:"
    line = '+' + '-' * (col_widths[0]+2) + '+' + '-' * (col_widths[1]+2) + '+' + '-' * (col_widths[2]+2) + '+'
    output.append(header)
    output.append(line)
    
    # Column headers
    output.append(f"| {'Module Name':{col_widths[0]}} | {'N_parameters':{col_widths[1]}} | {'Is Training':{col_widths[2]}} |")
    output.append(line)
    
    # Iterate through the first-level modules
    for name, module in model.named_children():
        # Calculate the number of parameters in the current module
        n_module_parameters = sum(p.numel() for p in module.parameters())
        
        # Check if all parameters in the module are frozen
        module_training = any(p.requires_grad for p in module.parameters())
        
        # Format the module details
        training_status = "Yes" if module_training else "No"
        formatted_params = f"{n_module_parameters:,}"
        output.append(f"| {name:{col_widths[0]}} | {formatted_params:>{col_widths[1]}} | {training_status:^{col_widths[2]}} |")
    
    # Append the final line
    output.append(line)
    
    # Print the entire table at once
    return "\n".join(output)

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

    if getattr(cfg.common, 'wandb_project', None) is not None and getattr(cfg.common, 'wandb_name', None) is not None:
        import wandb
        cfg.training.report_to = 'wandb'
        wandb.init(project=cfg.common.wandb_project, name=cfg.common.wandb_name)

    training_args = TrainingArguments(
        **cfg.training
    )

    # Counting parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Printing the information
    logger.info(f"Number of parameters: {total_params:,} ({trainable_params:,} Trained)")

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

    model = MultiModalEmbedderModel.build_model(cfg.model, train_dataset)

    logger.info(f"\n{model}\n")
    logger.info(f"\n{print_module_details(model)}\n")
    
    train_model(model, train_dataset, val_dataset, train_dataset.collate_fn, cfg)

    model.save_pretrained(cfg.training.output_dir + "/trained_model")
    train_dataset.src_tokenizer.save_pretrained(cfg.training.output_dir + "/trained_src_tokenizer")
    val_dataset.tgt_tokenizer.save_pretrained(cfg.training.output_dir + "/trained_tgt_tokenizer")

if __name__ == "__main__":
    main()

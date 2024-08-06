import torch
import torch.nn as nn
import argparse
import logging
import multiprocessing
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from omegaconf import OmegaConf, DictConfig

# Load custom modules (these need to be correctly defined in your project structure)
from multimodal_embedder.data.signwriting import SignWritingDataset
from multimodal_embedder.data.data_configs import MultimodalMTDataConfig
from multimodal_embedder.data.utils import _transform
from multimodal_embedder.models import MultiModalEmbedderModel

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def log_environment(training_args):
    n_cpus = multiprocessing.cpu_count()
    n_workers = training_args.dataloader_num_workers
    logger.info("***********************CUDA environments for all %d workers***********************", n_workers)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        for i in range(n_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info("GPU %d: capabilities = %s; total memory = %.1f GB; name = %s",
                        i, str(gpu_props.major) + '.' + str(gpu_props.minor), gpu_props.total_memory / 1e9, gpu_props.name)
    logger.info("***********************Environment setup complete for all %d CPU workers***********************", n_workers)
    if n_workers != n_cpus:
        logger.warning(f"Specify a number of workers that match with the availabel ones for an optimal load of the data. {n_workers} workers were specified, while {n_cpus} were available.")
    n_gpus = torch.cuda.device_count()
    logger.info("Training on %d devices (GPUs/TPUs)", n_gpus)
    logger.info("Batch size per device is %d", training_args.per_device_train_batch_size)
    logger.info("Gradient is accumulated for %d steps", training_args.gradient_accumulation_steps)
    effective_batch_size = n_gpus * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps if n_gpus > 0 else per_device_train_batch_size * gradient_accumulation_steps
    logger.info("Effective Batch Size is %d", effective_batch_size)

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
        cfg.training.run_name = cfg.common.wandb_name
        cfg.training.report_to = 'wandb'
        wandb.init(project=cfg.common.wandb_project, name=cfg.common.wandb_name)

    training_args = TrainingArguments(
        **cfg.training
    )
    
    # Additional logs
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {total_params:,} ({trainable_params:,} Trained)")
    log_environment(training_args)

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

    if getattr(cfg.training, 'remove_unused_columns', None):
        cfg.data.remove_unused_columns = cfg.training.remove_unused_columns
    
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

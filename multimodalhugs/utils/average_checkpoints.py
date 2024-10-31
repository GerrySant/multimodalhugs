"""
average_checkpoints.py

This script performs checkpoint averaging for models by incrementally loading 
and averaging model parameters from multiple checkpoint files. It creates a new directory with all 
the files from one of the original checkpoint directories, except that the `model.safetensors` file 
is replaced by the averaged checkpoint.

Usage:
    python average_checkpoints.py --output_dir <output_dir> --checkpoints <checkpoint_1> <checkpoint_2> ... <checkpoint_n>

Arguments:
    --output_dir   (str) : The directory where the averaged checkpoint will be saved.
                           The directory will contain all files from the first checkpoint 
                           directory, except that `model.safetensors` will be replaced by 
                           the new averaged model weights.
    --checkpoints  (list): List of paths to the `.safetensors` checkpoint files to be averaged.
                           The script will average the model weights across all specified 
                           checkpoint files.

Example:
    python average_checkpoints.py --output_dir /home/gsantm/store/multimodalhugs_experiments/hebrew_multimodalhugs/hebrew_multimodalhugs/checkpoint-avg \
                                  --verbose \
                                  --checkpoints /path/to/checkpoint1/model.safetensors \
                                               /path/to/checkpoint2/model.safetensors \
                                               /path/to/checkpoint3/model.safetensors

This example will create a new directory named `checkpoint-avg` inside the specified output path.
The averaged `model.safetensors` will be saved in this new directory.
"""

import os
import shutil
import argparse
import torch
from safetensors.torch import load_file, save_file, safe_open

def average_safetensors_checkpoints(ckpt_paths, output_dir, verbose=False):
    """
    Averages model weights from a list of checkpoint paths (.safetensors files) in a memory-efficient way
    and saves the averaged checkpoint.

    Parameters:
        ckpt_paths (list of str): List of paths to the checkpoint files to be averaged.
        output_dir (str): Directory to save the new averaged checkpoint.
        verbose (bool): If True, print detailed logs.
    """
    if not ckpt_paths:
        raise ValueError("No checkpoint paths provided.")
    
    # Initialize an empty dictionary to store the sum of parameters
    avg_state_dict = None
    num_checkpoints = len(ckpt_paths)
    
    if verbose:
        print(f"Number of checkpoints to average: {num_checkpoints}")
        print("Starting to load and average checkpoints...")

    # Loop over each checkpoint and accumulate parameter-wise sums
    for idx, path in enumerate(ckpt_paths):
        if verbose:
            print(f"Loading checkpoint {idx+1}/{num_checkpoints}: {path}")

        # Using safe_open to load the checkpoint
        with safe_open(path, framework="pt", device=0) as f:
            if avg_state_dict is None:
                # Initialize avg_state_dict with zeros using the structure of the first checkpoint
                if verbose:
                    print("Initializing average state dictionary...")
                avg_state_dict = {k: torch.zeros_like(f.get_tensor(k)) for k in f.keys()}
            
            for key in f.keys():
                avg_state_dict[key] += f.get_tensor(key) / num_checkpoints  # Accumulate the average incrementally

        if verbose:
            print(f"Finished processing checkpoint {idx+1}/{num_checkpoints}")

    if verbose:
        print("All checkpoints loaded and averaged.")

    # Determine the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Output directory created: {output_dir}")
        print("Copying auxiliary files from the original checkpoint directory...")

    # Copy all files from one of the original checkpoint directories to the new directory
    original_dir = os.path.dirname(ckpt_paths[0])
    for item in os.listdir(original_dir):
        s = os.path.join(original_dir, item)
        d = os.path.join(output_dir, item)
        if os.path.isdir(s):
            if verbose:
                print(f"Copying directory: {s} to {d}")
            shutil.copytree(s, d, dirs_exist_ok=True)
        elif not item.endswith("model.safetensors"):  # Avoid copying the original model.safetensors
            if verbose:
                print(f"Copying file: {s} to {d}")
            shutil.copy2(s, d)
        else:
            if verbose:
                print(f"Skipping original model file: {s}")

    # Save the averaged model to the new directory
    averaged_model_path = os.path.join(output_dir, "model.safetensors")
    if verbose:
        print(f"Saving averaged model to: {averaged_model_path}")
    save_file(avg_state_dict, averaged_model_path, metadata={"format": "pt"})

    print(f"Averaged checkpoint saved at: {output_dir}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Average model checkpoints in a memory-efficient way and save to a new directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the averaged checkpoint.")
    parser.add_argument("--checkpoints", nargs='+', required=True, help="Paths to the checkpoint files to be averaged.")
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')

    # Parse arguments
    args = parser.parse_args()

    # Call the averaging function
    if not bool(".ckpt" in args.checkpoints[0]):
        average_safetensors_checkpoints(args.checkpoints, args.output_dir, verbose=args.verbose)
    else:
        raise NotImplementedError("Currently the script only supports the model.safetensors format.")

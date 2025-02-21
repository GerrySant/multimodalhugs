import os
import re
import multiprocessing

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from omegaconf import OmegaConf, DictConfig

def serialize_config(config):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True) 
    return config

def print_module_details(model):
    col_widths = [30, 17, 25]
    output = []
    
    # Header and line setup
    header = "Model Summary:"
    line = '+' + '-' * (col_widths[0]+2) + '+' + '-' * (col_widths[1]+2) + '+' + '-' * (col_widths[2]+2) + '+'
    output.append(header)
    output.append(line)
    
    # Column headers
    output.append(f"| {'Module Name':{col_widths[0]}} | {'N_parameters':{col_widths[1]}} | {'N_training_parameters':{col_widths[2]}} |")
    output.append(line)
    
    # Iterate through the first-level modules
    for name, module in model.named_children():
        # Calculate the total number of parameters
        n_total_parameters = sum(p.numel() for p in module.parameters())
        
        # Calculate the number of training parameters
        n_training_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Format the details
        formatted_total_params = f"{n_total_parameters:,}"
        formatted_training_params = f"{n_training_parameters:,}"
        output.append(f"| {name:{col_widths[0]}} | {formatted_total_params:>{col_widths[1]}} | {formatted_training_params:>{col_widths[2]}} |")
    
    # Append the final line
    output.append(line)
    
    # Print the entire table at once
    return "\n".join(output)

def add_argument_to_the_config(config_path: str, section: str, argument_name: str, argument_value):
    yaml = YAML()
    yaml.preserve_quotes = True
    # Load the YAML configuration preserving order and comments
    with open(config_path, 'r') as file:
        config = yaml.load(file)

    # Ensure the section exists while preserving order
    if section not in config:
        config[section] = CommentedMap()

    # Update or add the argument_name entry
    config[section][argument_name] = argument_value

    # Insert a blank line before the section key
    config.yaml_set_comment_before_after_key(section, before='\n')

    # Write the updated configuration back to the file, preserving order
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def reformat_yaml_file(config_path: str):
    """Reformat a YAML file so that:
       - There are no extra blank lines inside sections.
       - There is exactly one blank line between top-level sections.
    """
    # Read the entire file as text
    with open(config_path, 'r') as f:
        text = f.read()

    # Remove all blank lines completely.
    # (This removes any extra blank lines within sections.)
    lines = [line.rstrip() for line in text.splitlines() if line.strip() != '']

    # Rebuild the file with a single blank line before each top-level key (except the first).
    # We consider a top-level key to be any line that starts with a non-space and contains a colon.
    final_lines = []
    for i, line in enumerate(lines):
        if i > 0 and re.match(r'^\S+:\s*', line):
            # If the previous line is not already blank, add a blank line.
            if final_lines and final_lines[-1].strip() != '':
                final_lines.append('')
        final_lines.append(line)

    # Join the lines using the newline character
    formatted_text = "\n".join(final_lines) + "\n"

    # Write back the formatted text to the file
    with open(config_path, 'w') as f:
        f.write(formatted_text)

def get_num_proc():
    # SGE
    if "NSLOTS" in os.environ:
        return int(os.environ["NSLOTS"])
    # SLURM
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    # PBS/Torque
    if "PBS_NUM_PPN" in os.environ:
        return int(os.environ["PBS_NUM_PPN"])
    if "PBS_NODEFILE" in os.environ:
        try:
            with open(os.environ["PBS_NODEFILE"]) as f:
                return len(f.readlines())
        except Exception:
            pass
    # HTCondor
    if "NUM_CPUS" in os.environ:
        return int(os.environ["NUM_CPUS"])
    # LSF
    if "LSB_DJOB_NUMPROC" in os.environ:
        return int(os.environ["LSB_DJOB_NUMPROC"])
    if "LSB_MAX_NUM_PROCESSORS" in os.environ:
        return int(os.environ["LSB_MAX_NUM_PROCESSORS"])
    # Fallback: number of system cores
    return multiprocessing.cpu_count()

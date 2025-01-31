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

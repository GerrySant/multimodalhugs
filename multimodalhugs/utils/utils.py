from omegaconf import OmegaConf, DictConfig

def serialize_config(config):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True) 
    return config

def print_module_details(model):
    # Define column widths
    col_widths = [30, 17, 12]
    
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
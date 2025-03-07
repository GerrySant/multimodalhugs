from dataclasses import asdict, fields, is_dataclass
import dataclasses
from typing import List, TypeVar

from omegaconf import OmegaConf
from transformers import HfArgumentParser

T = TypeVar("T")

# -----------------------------
# Functions for managing YAML configuration
# -----------------------------
def merge_arguments(cmd_args: T, extra_args: T, command_arg_names: List[str], yaml_arg_keys: List[str]) -> T:
    """
    Merge command-line arguments and configuration arguments for any dataclass instance.

    This function merges the attributes of two dataclass instances of the same type,
    following these rules:
    
    1. Identify the fields in `cmd_args` that were NOT explicitly provided on the command line.
       These fields are assumed to still have their default values (i.e. names not in `command_arg_names`).
       
    2. For each field:
         - If the value in `cmd_args` differs from that in `extra_args` **and** the field was not
           explicitly set on the command line (i.e. it is in the default list), override the value in
           `cmd_args` with the value from `extra_args`.
         - Otherwise, keep the command-line value.

    Only fields listed in `yaml_arg_keys` will be considered for merging.

    Args:
        cmd_args (T): The dataclass instance populated from command-line arguments.
        extra_args (T): The dataclass instance populated from configuration (e.g. YAML).
        command_arg_names (List[str]): The names of the arguments that were explicitly set on the command line.
        yaml_arg_keys (List[str]): The names of the arguments present in the configuration.

    Returns:
        T: The merged dataclass instance with updated fields.
        
    Raises:
        ValueError: If either cmd_args or extra_args is not a dataclass instance.
    """
    if not (is_dataclass(cmd_args) and is_dataclass(extra_args)):
        raise ValueError("Both cmd_args and extra_args must be dataclass instances.")
    
    default_arguments = [f.name for f in fields(cmd_args) if f.name not in command_arg_names]
    
    for f in fields(cmd_args):
        field_name = f.name
        if field_name in yaml_arg_keys:
            cmd_value = getattr(cmd_args, field_name)
            cfg_value = getattr(extra_args, field_name)
            if cmd_value != cfg_value and field_name in default_arguments:
                setattr(cmd_args, field_name, cfg_value)
    
    return cmd_args

def construct_kwargs(obj, not_used_keys=None):
    """
    Constructs a dictionary of keyword arguments from a dataclass instance by comparing
    each field's current value to its default. Fields listed in not_used_keys are skipped.
    """
    if not_used_keys is None:
        not_used_keys = []
    kwargs = {}
    obj_dict = asdict(obj)

    for field_info in fields(obj):
        if field_info.name in not_used_keys:
            continue
        # Check if the field has a default factory.
        if field_info.default_factory is not dataclasses.MISSING:
            default_value = field_info.default_factory()
        else:
            default_value = field_info.default
        # Add field if its value differs from the default.
        if obj_dict[field_info.name] != default_value:
            kwargs[field_info.name] = obj_dict[field_info.name]
    
    return kwargs

def filter_config_keys(config_section: dict, dataclass_type) -> dict:
    """
    Filters the keys from a configuration section based on the valid fields of the provided dataclass.
    """
    valid_keys = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in config_section.items() if k in valid_keys}

def merge_config_and_command_args(config_path, class_type, section, _args, remaining_args):
    """
    Merges YAML configuration with command-line arguments for a given section.

    Args:
        config_path (str): Path to the YAML configuration file.
        class_type: The dataclass type for this configuration section.
        section (str): The key in the YAML configuration corresponding to this section.
        _args: The dataclass instance populated from the command-line.
        remaining_args (List[str]): Remaining command-line arguments.
    
    Returns:
        The updated dataclass instance with merged configuration values.
    """
    yaml_conf = OmegaConf.load(config_path)
    yaml_dict = OmegaConf.to_container(yaml_conf, resolve=True)
    _parser = HfArgumentParser((class_type,))
    filtered_yaml = filter_config_keys(yaml_dict[section], class_type)
    extra_args = _parser.parse_dict(filtered_yaml)[0]
    command_arg_names = [value[2:].replace("-", "_") for value in remaining_args if value.startswith('--')]
    yaml_keys = yaml_dict[section].keys()
    _args = merge_arguments(
        cmd_args=_args,
        extra_args=extra_args,
        command_arg_names=command_arg_names,
        yaml_arg_keys=yaml_keys
    )
    return _args

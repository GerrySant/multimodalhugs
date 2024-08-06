from omegaconf import OmegaConf, DictConfig

def serialize_config(config):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True) 
    return config
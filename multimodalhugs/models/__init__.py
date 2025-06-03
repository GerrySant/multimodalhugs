import importlib
import os
from .utils import *
from transformers import AutoConfig, AutoModelForSeq2SeqLM

# Get current directory (models/)
models_dir = os.path.dirname(__file__)

# Loop through all submodules in the models folder
for name in os.listdir(models_dir):
    subdir = os.path.join(models_dir, name)
    if os.path.isdir(subdir) and not name.startswith("__"):
        try:
            module = importlib.import_module(f"multimodalhugs.models.{name}")
            model_class = getattr(module, "MODEL_CLASS", None)
            config_class = getattr(module, "CONFIG_CLASS", None)
            config_name = getattr(module, "CONFIG_NAME", None)

            if model_class and config_class and config_name:
                AutoConfig.register(config_name, config_class)
                AutoModelForSeq2SeqLM.register(config_class, model_class)
        except Exception as e:
            print(f"Could not register model in '{name}': {e}")





# from .utils import *
# from .multimodal_embedder import MultiModalEmbedderModel, MultiModalEmbedderConfig
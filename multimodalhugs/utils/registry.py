MODEL_REGISTRY = {}
DATASET_REGISTRY = {}

def register_model(model_type):
    """Decorator to register a model class under a model type."""
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator

def get_model_class(model_type):
    """Retrieve a model class from the registry by model type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]

def register_dataset(dataset_type):
    """Decorator to register a dataset class under a dataset type."""
    def decorator(cls):
        DATASET_REGISTRY[dataset_type] = cls
        return cls
    return decorator

def get_dataset_class(dataset_type):
    """Retrieve a dataset class from the registry by dataset type."""
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {dataset_type}. "
                         f"Available datasets: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_type]

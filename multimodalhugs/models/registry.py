MODEL_REGISTRY = {}

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
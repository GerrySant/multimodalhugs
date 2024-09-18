import torch
import torch.nn as nn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim, factor, layernorm_before=False):
        super(Adapter, self).__init__()
        self.layernorm_before = layernorm_before
        if input_dim != output_dim:
            logger.info(f"Detected Adapter's input/output of different dimension ({input_dim}/{output_dim}), a projection layer will be used")
            self.projection_layer = nn.Linear(in_features=input_dim, out_features=output_dim)
            input_dim = output_dim
        else:
            self.projection_layer = None
        if layernorm_before:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True)
        self.expand_linear = nn.Linear(in_features=input_dim, out_features=input_dim * factor)
        self.relu = nn.ReLU()
        self.shrink_linear = nn.Linear(in_features=input_dim * factor, out_features=output_dim)
        if not layernorm_before:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True)


    def forward(self, x):
        # Shape of x: [T, B, C]
        if self.projection_layer is not None:
            x = self.projection_layer(x)
        if self.layernorm_before:
            out = self.layer_norm(x)
            out = self.expand_linear(out)
        else:
            out = self.expand_linear(x)
        out = self.relu(out)
        out = self.shrink_linear(out)
        if not self.layernorm_before:
            out = self.layer_norm(out)
        # Residual connection
        if x.shape == out.shape:
            out = x + out
        return out

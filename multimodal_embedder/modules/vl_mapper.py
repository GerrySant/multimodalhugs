import torch
import torch.nn as nn
from multimodal_embedder.modules import Adapter

class VLMapper(nn.Module):
    def __init__(self, feat_dim, output_dim, mapping_layer_type, layer_norm_before,
                 adapter_factor=None, p_dropout=None, layer_norm=None, activation=None):
        super(VLMapper, self).__init__()

        if layer_norm_before and mapping_layer_type != 'adapter':
            self.layer_norm_before = nn.LayerNorm(feat_dim)
        else:
            self.layer_norm_before = None

        # Definition of mapping_layer
        if mapping_layer_type == 'adapter':
            self.mapping_layer = Adapter(
                input_dim=feat_dim,
                output_dim=output_dim,
                factor=adapter_factor,
                layernorm_before=layer_norm_before
            )
        elif mapping_layer_type == 'linear':
            self.mapping_layer = nn.Linear(feat_dim, output_dim)
        else:
            self.mapping_layer = None

        if p_dropout is not None and not isinstance(self.mapping_layer, Adapter):
            self.dropout = nn.Dropout(p=p_dropout)
        else:
            self.dropout = None

        if layer_norm and not isinstance(self.mapping_layer, Adapter):
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        if activation and not isinstance(self.mapping_layer, Adapter):
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def forward(self, x):
        if self.layer_norm_before is not None:
            x = self.layer_norm_before(x)
        
        if self.mapping_layer is not None:
            x = self.mapping_layer(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        return x

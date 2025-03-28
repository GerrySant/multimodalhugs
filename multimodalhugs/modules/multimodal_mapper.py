import torch
import torch.nn as nn
from multimodalhugs.modules import Adapter, CNNAdapter

class MultimodalMapper(nn.Module):
    def __init__(self, feat_dim, output_dim, mapping_layer_type, layer_norm_before,
                 adapter_factor=None, adapter_ksize=None, adapter_stride=None, 
                 p_dropout=None, layer_norm=None, activation=None):
        super(MultimodalMapper, self).__init__()

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

        if mapping_layer_type == 'cnn_adapter':
            kwargs = {}
            if adapter_ksize is not None:
                kwargs["kernel_sizes"] = adapter_ksize
            if adapter_stride is not None:
                kwargs["strides"] = adapter_stride
            self.mapping_layer = CNNAdapter(
                input_dim=feat_dim,
                output_dim=output_dim,
                factor=adapter_factor,
                **kwargs
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

    def forward(self, x, mask):
        if self.layer_norm_before is not None:
            x = self.layer_norm_before(x)
        
        if self.mapping_layer is not None:
            if isinstance(self.mapping_layer, CNNAdapter):
                x, mask = self.mapping_layer(x, mask)
            else:
                x = self.mapping_layer(x)

        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        return x, mask

    def mask_correction(self, mask):
        if isinstance(self.mapping_layer, CNNAdapter):
            return self.mapping_layer.get_out_mask_tensor(mask)
        return mask
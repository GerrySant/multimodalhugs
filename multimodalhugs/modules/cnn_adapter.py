import torch
import torch.nn as nn
import logging
from typing import List, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNAdapter(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        factor: int, 
        kernel_sizes: List[int] = [3, 3], 
        strides: Union[int, List[int]] = [2, 2]
    ):
        """
        1D Convolutional subsampler that expands and compresses feature representations.

        Args:
            input_dim (int): Number of input channels.
            output_dim (int): Number of output channels.
            factor (int): Expansion factor for the intermediate representation.
            kernel_sizes (List[int]): List of kernel sizes for each convolutional layer.
            strides (int or List[int]): Stride(s) for each convolutional layer. If int, same stride is applied to all.
        """
        super(CNNAdapter, self).__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = (kernel_sizes,)
        if isinstance(strides, int):
            strides = (strides,)

        if input_dim != output_dim:
            logger.info(f" Adapter's input/output dimensions differ ({input_dim}/{output_dim}), applying projection layer.")
            self.projection_layer = nn.Linear(in_features=input_dim, out_features=output_dim)
            input_dim = output_dim
        else:
            self.projection_layer = None

        self.conv_layers = nn.ModuleList()
        in_channels = input_dim

        # Ensure strides is a list matching kernel_sizes
        if isinstance(strides, int):
            strides = [strides] * len(kernel_sizes)
        elif len(strides) != len(kernel_sizes):
            raise ValueError("Length of `strides` must match `kernel_sizes`.")

        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, strides)):
            out_channels = output_dim * 2 if i == len(kernel_sizes) - 1 else output_dim * factor
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
                )
            )
            in_channels = out_channels // 2  # GLU halves the channels

        self.relu = nn.ReLU()
        self.strides = strides  # Store strides for sequence length correction

    def get_out_mask_tensor(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the new boolean mask after applying Conv1D subsampling.

        Args:
            mask (torch.Tensor): Original mask of shape (B, T) with 1s for valid tokens and 0s for padding.

        Returns:
            torch.Tensor: New mask of shape (B, T') after subsampling.
        """
        # Step 1: Get lengths from input mask
        src_lengths = mask.sum(dim=1)

        # Step 2: Compute new lengths using strides
        new_lengths = src_lengths.clone()
        for stride in self.strides:
            new_lengths = ((new_lengths.float() - 1) / stride + 1).floor().long()

        # Step 3: Convert lengths back to mask
        max_len = new_lengths.max().item()
        new_mask = torch.arange(max_len, device=mask.device).expand(mask.size(0), max_len) < new_lengths.unsqueeze(1)
        return new_mask.to(mask.dtype)  # e.g., int32 or bool, matching original

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass of the subsampler.

        Args:
            x (torch.Tensor): Input tensor of shape (T, B, C).
            mask (torch.Tensor): Original lengths mask.

        Returns:
            torch.Tensor: Processed tensor.
            torch.Tensor: Adjusted lengths mask.
        """
        x = x.transpose(0, 1)  # (T, B, C)
        if self.projection_layer is not None:
            x = self.projection_layer(x)

        x = x.permute(1, 2, 0)  # Convert to (B, C, T) for Conv1D

        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)  # Apply Gated Linear Unit (GLU)

        x = x.permute(0, 2, 1)  # (B, T', C)

        return x, self.get_out_mask_tensor(mask)

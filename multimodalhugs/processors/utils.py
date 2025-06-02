import torch

def frame_skipping(x: torch.Tensor, t_dim: int, stride: int) -> torch.Tensor:
    """
    Downsample the temporal dimension of a tensor by a fixed stride.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        t_dim (int): Index of the temporal dimension.
        stride (int): Sampling stride (keep every `stride`-th element).

    Returns:
        torch.Tensor: Tensor with the same shape except the temporal dimension
                      is reduced to ceil(orig_length/stride).
    """
    # build slice objects for all dims, stepping the temporal dim by `stride`
    slices = [slice(None)] * x.dim()
    slices[t_dim] = slice(None, None, stride)
    return x[tuple(slices)]
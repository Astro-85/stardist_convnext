# Built on top of https://github.com/pvjosue/pytorch_convNd

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable
import math

from typing import List, Tuple, Callable, Union


class UpsampleNd(nn.Module):
    """
    An n-dimensional upsampling layer.

    For 1D, 2D, and 3D upsampling, this module falls back to the highly optimized
    native PyTorch implementation (nn.Upsample), which wraps F.interpolate.

    For dimensions greater than 3, it uses a custom implementation that currently
    only supports 'nearest' neighbor mode. This is functionally equivalent to how
    F.interpolate would behave for >3D inputs if it were implemented.
    """
    def __init__(self,
                 num_dims: int,
                 size: Tuple[int, ...] = None,
                 scale_factor: Tuple[float, ...] = None,
                 mode: str = 'nearest',
                 align_corners: bool = None):
        """
        Initializes the UpsampleNd module.

        Args:
            num_dims (int): The number of spatial dimensions.
            size (Tuple[int, ...], optional): The target output size for the spatial dimensions.
                                              Defaults to None.
            scale_factor (Tuple[float, ...], optional): The multiplier for the spatial size.
                                                        Defaults to None.
            mode (str, optional): The upsampling algorithm. For num_dims > 3, only
                                  'nearest' is supported. Defaults to 'nearest'.
            align_corners (bool, optional): For certain modes, specifies how to handle
                                            corner pixels. See PyTorch documentation.
                                            Defaults to None.
        """
        super().__init__()

        # ---------------------------------------------------------------------
        # Assertions and argument storage
        # ---------------------------------------------------------------------
        if (size is None) == (scale_factor is None):
            raise ValueError("Exactly one of size or scale_factor must be specified")

        self.num_dims = num_dims
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
        self.native_upsample = None

        # ---------------------------------------------------------------------
        # NATIVE FALLBACK: Use standard Pytorch layer for 1D, 2D, 3D
        # ---------------------------------------------------------------------
        if self.num_dims <= 3:
            # nn.Upsample is a wrapper for F.interpolate and handles all modes
            self.native_upsample = nn.Upsample(size=self.size,
                                               scale_factor=tuple(self.scale_factor),
                                               mode=self.mode,
                                               align_corners=self.align_corners)
        # ---------------------------------------------------------------------
        # CUSTOM IMPLEMENTATION: For dimensions > 3
        # ---------------------------------------------------------------------
        else:
            if self.mode != 'nearest':
                raise NotImplementedError(
                    f"For dimensions > 3, UpsampleNd only supports 'nearest' mode, but got '{self.mode}'.")
            
            # Normalize scale_factor to a tuple if it's a single number
            if self.scale_factor is not None and not isinstance(self.scale_factor, (list, tuple)):
                self.scale_factor = tuple(self.scale_factor for _ in range(self.num_dims))
            
            # Normalize size to a tuple if it's a single number
            if self.size is not None and not isinstance(self.size, (list, tuple)):
                self.size = tuple(self.size for _ in range(self.num_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the n-dimensional upsampling.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D1, D2, ..., Dn),
                              where n is num_dims.

        Returns:
            torch.Tensor: The upsampled tensor.
        """
        # If we are using the native Pytorch layer, just call it.
        if self.native_upsample is not None:
            return self.native_upsample(x)

        # Custom implementation for >3D 'nearest' upsampling.
        # This logic mimics F.interpolate(..., mode='nearest').
        
        input_rank = x.dim()
        if input_rank - 2 != self.num_dims:
            raise ValueError(f"Input tensor has {input_rank - 2} spatial dimensions, "
                             f"but layer was configured for {self.num_dims} dimensions.")
        
        input_spatial_shape = x.shape[2:]

        # 1. Determine the target output spatial shape
        if self.size is not None:
            if len(self.size) != self.num_dims:
                raise ValueError(f"size tuple must have length {self.num_dims}, but got {len(self.size)}")
            target_spatial_shape = self.size
        else:  # self.scale_factor must be non-None
            if len(self.scale_factor) != self.num_dims:
                 raise ValueError(f"scale_factor tuple must have length {self.num_dims}, "
                                  f"but got {len(self.scale_factor)}")
            target_spatial_shape = [int(s * sf) for s, sf in zip(input_spatial_shape, self.scale_factor)]

        # 2. For each spatial dimension, compute the source indices
        indices = []
        for i in range(self.num_dims):
            in_dim = input_spatial_shape[i]
            out_dim = target_spatial_shape[i]
            
            # This is the core of 'nearest' upsampling: map output coordinate `j` to input coordinate `i`.
            # i = floor(j / scale_factor) which is equivalent to floor(j * (in_dim / out_dim))
            src_coords = torch.arange(out_dim, device=x.device, dtype=torch.float32)
            src_indices = torch.floor(src_coords * (in_dim / out_dim)).long()
            
            # Clamp indices to be safe, although floor should not exceed in_dim-1
            src_indices = torch.clamp(src_indices, 0, in_dim - 1)
            
            indices.append(src_indices)
        
        # 3. Use meshgrid to create a grid of indices for all spatial dimensions
        # 'ij' indexing ensures that the first index tensor varies in the first output dimension, etc.
        # This matches the tensor memory layout.
        grid = torch.meshgrid(indices, indexing='ij')
        
        # 4. Index the input tensor.
        # The grid provides the indices for the spatial dimensions.
        # The first two dimensions (batch and channel) are taken as a whole (slice(None)).
        return x[(slice(None), slice(None)) + grid]
        

class ConvNd(nn.Module):
    """
    An n-dimensional convolutional layer.

    For 1D, 2D, and 3D convolutions, this module falls back to the highly optimized
    native PyTorch implementations (nn.Conv1d, nn.Conv2d, nn.Conv3d).

    For dimensions greater than 3, it uses a recursive approach, building an
    n-dimensional convolution from a series of (n-1)-dimensional convolutions.
    This recursive implementation is functional but significantly slower.
    """
    def __init__(self,in_channels: int,
                 out_channels: int,
                 num_dims: int,
                 kernel_size,
                 stride,
                 padding,
                 is_transposed = False,
                 padding_mode = 'zeros',
                 output_padding = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 rank: int = 0,
                 use_bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):
        super(ConvNd, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions and argument normalization
        # ---------------------------------------------------------------------
        if not isinstance(kernel_size, (Tuple, List)):
            kernel_size = tuple(kernel_size for _ in range(num_dims))
        if not isinstance(stride, (Tuple, List)):
            stride = tuple(stride for _ in range(num_dims))
        if not isinstance(padding, (Tuple, List)):
            padding = tuple(padding for _ in range(num_dims))
        if not isinstance(output_padding, (Tuple, List)):
            output_padding = tuple(output_padding for _ in range(num_dims))
        if not isinstance(dilation, (Tuple, List)):
            dilation = tuple(dilation for _ in range(num_dims))

        assert len(kernel_size) == num_dims, 'nD kernel size expected!'
        assert len(stride) == num_dims, 'nD stride size expected!'
        assert len(padding) == num_dims, 'nD padding size expected!'
        assert len(output_padding) == num_dims, 'nD output_padding size expected!'
        assert len(dilation) == num_dims, 'nD dilation size expected!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.rank = rank
        self.is_transposed = is_transposed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dims = num_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # Set a default kernel initializer if none is provided (for top-level call)
        if rank == 0 and self.kernel_initializer is None:
            # Kaiming uniform sqrt(5) is the default for standard conv layers.
            # The original code used a different uniform. We will keep it for consistency.
            k = 1.0 / math.sqrt(self.in_channels * torch.prod(torch.tensor(self.kernel_size, dtype=torch.float32)))
            self.kernel_initializer = lambda x: torch.nn.init.uniform_(x, -k, k)

        self.native_conv = None

        # ---------------------------------------------------------------------
        # NATIVE FALLBACK: Use standard Pytorch layers for 1D, 2D, 3D
        # ---------------------------------------------------------------------
        if self.num_dims <= 3:
            if self.is_transposed:
                conv_class = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[self.num_dims - 1]
            else:
                conv_class = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[self.num_dims - 1]

            native_conv_args = {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "dilation": self.dilation,
                "groups": self.groups,
                "bias": self.use_bias,
                "padding_mode": self.padding_mode,
            }
            # output_padding is only a valid argument for transposed convolutions
            if self.is_transposed:
                native_conv_args["output_padding"] = self.output_padding
            
            self.native_conv = conv_class(**native_conv_args)

            # Apply custom initializers if provided
            if self.kernel_initializer is not None:
                self.kernel_initializer(self.native_conv.weight)
            if self.use_bias and self.bias_initializer is not None:
                self.bias_initializer(self.native_conv.bias)

        # ---------------------------------------------------------------------
        # RECURSIVE IMPLEMENTATION: For dimensions > 3
        # ---------------------------------------------------------------------
        else:
            # Dilation > 1 is not implemented for the recursive version.
            assert all(d == 1 for d in self.dilation), \
                'Dilation rate other than 1 not yet implemented for num_dims > 3!'

            # For recursive calls, the base convolution is always 3D
            max_dims = 3
            if is_transposed:
                self.conv_f = nn.ConvTranspose3d
            else:
                self.conv_f = nn.Conv3d

            # Manually create bias parameter for the top-level recursive layer
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros(out_channels))
                if self.bias_initializer is not None:
                    self.bias_initializer(self.bias)
            else:
                self.register_parameter('bias', None)

            # Use a ModuleList to store layers to make them trainable
            self.conv_layers = torch.nn.ModuleList()

            next_dim_len = self.kernel_size[0]
            
            for _ in range(next_dim_len):
                if self.num_dims - 1 > max_dims:
                    # Initialize a Conv_n-1_D layer recursively
                    conv_layer = convNd(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        use_bias=self.use_bias, # Propagate bias setting
                                        num_dims=self.num_dims - 1,
                                        rank=self.rank - 1,
                                        is_transposed=self.is_transposed,
                                        kernel_size=self.kernel_size[1:],
                                        stride=self.stride[1:],
                                        groups=self.groups,
                                        dilation=self.dilation[1:],
                                        padding=self.padding[1:],
                                        padding_mode=self.padding_mode,
                                        output_padding=self.output_padding[1:],
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer)
                else:
                    # Base case: Initialize a 3D Conv layer
                    # Bias is disabled in sub-layers as it's added once at the end.
                    conv_layer = self.conv_f(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             bias=False,
                                             kernel_size=self.kernel_size[1:],
                                             dilation=self.dilation[1:],
                                             stride=self.stride[1:],
                                             padding=self.padding[1:],
                                             padding_mode=self.padding_mode,
                                             groups=self.groups)
                    if self.is_transposed:
                        conv_layer.output_padding = self.output_padding[1:]

                    # Apply initializer functions to weight tensor
                    if self.kernel_initializer is not None:
                        self.kernel_initializer(conv_layer.weight)

                self.conv_layers.append(conv_layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If we are using the native Pytorch layer, just call it.
        if self.native_conv is not None:
            return self.native_conv(input)

        # Otherwise, use the slower recursive implementation for >3D.
        # Pad the input if is not transposed convolution
        if not self.is_transposed:
            padding = list(self.padding)
            # Pad input if this is the parent convolution ie rank=0
            if self.rank == 0:
                inputShape = list(input.shape)
                inputShape[2] += 2 * self.padding[0]
                padSize = (0, 0, self.padding[0], self.padding[0])
                padding[0] = 0
                if self.padding_mode == 'zeros':
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize, 'constant', 0).view(inputShape)
                else:
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize, self.padding_mode).view(inputShape)

        # Define shortcut names for dimensions of input and kernel
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size

        if not self.is_transposed:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple([math.floor((size_i[x] + 2 * padding[x] - size_k[x]) / self.stride[x] + 1) for x in range(len(size_i))])
            # Compute size of the output without stride
            size_ons = tuple([size_i[x] - size_k[x] + 1 for x in range(len(size_i))])
        else:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple([(size_i[x] - 1) * self.stride[x] - 2 * self.padding[x] + (size_k[x] - 1) + 1 + self.output_padding[x] for x in range(len(size_i))])

        # Output tensors for each 3D frame
        frame_results = size_o[0] * [torch.zeros((b, self.out_channels) + size_o[1:], device=input.device)]
        empty_frames = size_o[0] * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(size_k[0]):
            # iterate inputs first dimension
            for j in range(size_i[0]):

                # Add results to this output frame
                if self.is_transposed:
                    out_frame = i + j * self.stride[0] - self.padding[0]
                else:
                    out_frame = j - (i - size_k[0] // 2) - (size_i[0] - size_ons[0]) // 2 - (1 - size_k[0] % 2)
                    k_center_position = out_frame % self.stride[0]
                    out_frame = math.floor(out_frame / self.stride[0])
                    if k_center_position != 0:
                        continue
                
                if out_frame < 0 or out_frame >= size_o[0]:
                    continue

                # Prepare input for next dimension
                conv_input = input.view(b, c_i, size_i[0], -1)
                conv_input = conv_input[:, :, j, :].view((b, c_i) + size_i[1:])

                # Convolve
                frame_conv = self.conv_layers[i](conv_input)

                if empty_frames[out_frame] is None:
                    frame_results[out_frame] = frame_conv
                    empty_frames[out_frame] = 1
                else:
                    frame_results[out_frame] += frame_conv

        result = torch.stack(frame_results, dim=2)

        if self.use_bias:
            # Add bias term. Reshaping is a robust way to broadcast.
            bias_shape = [1, self.out_channels] + [1] * (self.num_dims)
            result = result + self.bias.view(bias_shape)
        
        return result
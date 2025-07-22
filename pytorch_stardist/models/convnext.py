# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from typing import List, NamedTuple, Optional, Tuple, Type, Union

from .convNd import ConvNd, UpsampleNd


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        ch_dim (int): Number of input channels.
        num_dims (int): Number of dimensions in the input tensor.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, ch_dim, num_dims, kernel_size, stride=None, padding=None, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.num_dims = num_dims
        self.Conv = ConvNd
        self.dwconv = self.Conv(ch_dim, ch_dim, num_dims, kernel_size, stride, padding, groups=ch_dim) # depthwise conv
        self.norm = LayerNorm(ch_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(ch_dim, 2 * ch_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * ch_dim, ch_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((ch_dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # Generalized permute: (N, C, D1, D2, ..., Dn) -> (N, D1, D2, ..., Dn, C)
        # Create permutation order: [0, 2, 3, ..., num_dims+1, 1]
        perm_to_channels_last = [0] + list(range(2, self.num_dims + 2)) + [1]
        x = x.permute(*perm_to_channels_last)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # Generalized permute back: (N, D1, D2, ..., Dn, C) -> (N, C, D1, D2, ..., Dn)
        # Create permutation order: [0, num_dims+1, 1, 2, ..., num_dims]
        perm_to_channels_first = [0, self.num_dims + 1] + list(range(1, self.num_dims + 1))
        x = x.permute(*perm_to_channels_first)

        x = input + self.drop_path(x)
        return x


class ConvNeXtEncoder(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        grid (list[int]): Grid size for the stem and downsampling layers. Default: [1, 4, 4, 4]
        out_indices (list[int): Output indices of the feature maps. Default: [0, 1, 2, 3]
    """
    def __init__(
        self, 
        in_chans=1, 
        num_dims=3, 
        depths=[3, 3, 9, 3], 
        ch_dims=[96, 192, 384, 768], 
        stem_grid=[4, 4, 4], 
        downsample_grid=[2, 2, 2],
        kernel_size=[7, 7, 7],
        stride=[1, 1, 1],
        padding=[3, 3, 3],
        drop_path_rate=0., 
        layer_scale_init_value=1e-6, 
        out_indices=[0, 1, 2]
    ):
        super().__init__()
        self.Conv = ConvNd
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            self.Conv(in_chans, ch_dims[0], num_dims, kernel_size=stem_grid, stride=stem_grid, padding=0),
            LayerNorm(ch_dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    self.Conv(ch_dims[i], ch_dims[i+1], num_dims, kernel_size=downsample_grid, stride=downsample_grid, padding=0),
                    LayerNorm(ch_dims[i+1], eps=1e-6, data_format="channels_first"),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(ch_dims[i], num_dims, kernel_size, stride, padding, drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i in range(4):
            if i in self.out_indices:
                layer = norm_layer(ch_dims[i])
                layer_name = f'norm{i}'
                self.add_module(layer_name, layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, self.Conv):
            if m.native_conv is not None:
                trunc_normal_(m.native_conv.weight, std=.02)
                nn.init.constant_(m.native_conv.bias, 0)
            else:
                for conv_layer in m.conv_layers:
                    trunc_normal_(conv_layer.weight, std=.02)
                    if conv_layer.bias is not None:
                        nn.init.constant_(conv_layer.bias, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)
    

class ConvNeXtDecoder(nn.Module):
    def __init__(
        self, 
        in_chans=768,
        num_dims=3, 
        depths=[3, 9, 3, 3], 
        ch_dims=[384, 192, 96, 96], 
        upsample_grid=[1, 2, 2, 2],
        kernel_size=[7, 7, 7],
        stride=[1, 1, 1],
        padding=[3, 3, 3],
        drop_path_rate=0., 
        layer_scale_init_value=1e-6, 
        out_indices=[0, 1, 2]
    ):
        super().__init__()
        self.Conv = ConvNd
        self.Upsample = UpsampleNd
        self.upsample_layers = nn.ModuleList()
        for i in range(4):
            upsample_layer = nn.Sequential(
                    self.Upsample(num_dims, scale_factor=upsample_grid, mode='nearest'),
                    self.Conv(([in_chans]+ch_dims)[i], ch_dims[i], num_dims, kernel_size, stride, padding),
                    LayerNorm(ch_dims[i], eps=1e-6, data_format="channels_first"),
            )
            self.upsample_layers.append(upsample_layer)

        self.skip_layers = nn.ModuleList() # skip connections for the upsampled feature maps
        for i in range(3):
            skip_layer = nn.Sequential(
                self.Conv(2*ch_dims[i], ch_dims[i], num_dims, kernel_size, stride, padding),
                LayerNorm(ch_dims[i], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.skip_layers.append(skip_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(ch_dims[i], num_dims, kernel_size, stride, padding, drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i in range(4):
            if i in self.out_indices:
                layer = norm_layer(ch_dims[i])
                layer_name = f'norm{i}'
                self.add_module(layer_name, layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, self.Conv):
            if m.native_conv is not None:
                trunc_normal_(m.native_conv.weight, std=.02)
                nn.init.constant_(m.native_conv.bias, 0)
            else:
                for conv_layer in m.conv_layers:
                    trunc_normal_(conv_layer.weight, std=.02)
                    if conv_layer.bias is not None:
                        nn.init.constant_(conv_layer.bias, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]):
        outs = []
        skips = features[:-1][::-1]
        x = features[-1]
        for i in range(4):
            x = self.upsample_layers[i](x)
            if i <= 2:
                x = torch.cat([x, skips[i]], dim=1)
                x = self.skip_layers[i](x) 
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
            
        return tuple(outs)
            

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # Create broadcasting shape: [C, 1, 1, ..., 1] with (x.dim() - 1) ones
            weight_shape = [self.weight.shape[0]] + [1] * (x.dim() - 2)
            bias_shape = [self.bias.shape[0]] + [1] * (x.dim() - 2)
            x = self.weight.view(weight_shape) * x + self.bias.view(bias_shape)
            return x
import itertools
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

from .convnext import ConvNeXtEncoder, ConvNeXtDecoder


######################################################################
#                    BaseNetwork and helper
######################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


class BaseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

def define_stardist_net(opt):
    net = StarDistConvnextUnet(config=opt)
    return net


######################################################################
#                           ConvNext-UNet
######################################################################

class StarDistConvnextUnet(BaseNetwork):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = ConvNeXtEncoder(
            in_chans=config.n_channel_in, 
            num_dims=config.n_dim,
            depths=config.convnext_encoder_depths,
            ch_dims=config.convnext_encoder_channels,
            stem_grid=config.grid,
            downsample_grid=config.convnext_encoder_downsample,
            kernel_size=config.convnext_encoder_kernel_size,
            stride=config.convnext_encoder_stride,
            padding=config.convnext_encoder_padding,
            out_indices=[0, 1, 2, 3]
        )

        self.decoder = ConvNeXtDecoder(
            in_chans=config.convnext_encoder_channels[-1],
            num_dims=config.n_dim,
            depths=config.convnext_decoder_depths,
            ch_dims=config.convnext_decoder_channels,
            upsample_grid=config.convnext_decoder_upsample,
            kernel_size=config.convnext_decoder_kernel_size,
            stride=config.convnext_decoder_stride,
            padding=config.convnext_decoder_padding,
            out_indices=[3]
        )
        
        n_filter_base = self.config.convnext_decoder_channels[-1]
        
        if self.config.net_conv_after_unet > 0:
            self.final_layer = nn.Sequential(
                nn.Conv3d(n_filter_base, self.config.net_conv_after_unet,
                     self.config.net_conv_after_unet_kernel_size, padding="same"),
                nn.GELU()
            )
            final_layer_channels = self.config.net_conv_after_unet
        else:
            self.final_layer = Identity()
            final_layer_channels = n_filter_base
        
        self.output_prob = nn.Conv3d(final_layer_channels, 1, kernel_size=2, stride=2)
        self.output_dist = nn.Conv3d(final_layer_channels, self.config.n_rays, kernel_size=2, stride=2)

        if self.config.n_classes is not None:
            self.output_prob_classes = nn.Sequential(
                nn.Conv3d(final_layer_channels, self.config.n_classes + 1, kernel_size=2, stride=2),
                nn.Softmax()
            )
        
    def forward(self, x): 
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)

        if self.config.n_dim == 4:
            final_out = torch.mean(decoder_out[-1], dim=2)
        elif self.config.n_dim == 3:
            final_out = decoder_out[-1]
        else:
            raise NotImplementedError(f"Unsupported number of dimensions: {self.config.n_dim}")
        
        final_out = self.final_layer(final_out)

        if self.config.n_classes is not None:
            output_classes = self.output_prob_classes(final_out)
        else:
            output_classes = None       

        return self.output_dist(final_out), self.output_prob(final_out), output_classes
    
    def predict(self, x):
        rays, prob, class_prob = self.forward(x)
        prob = torch.sigmoid(prob)
        if class_prob is not None:
            class_prob = torch.nn.functional.softmax(class_prob, dim=-(1 + self.n_dim))
        return rays, prob, class_prob

    @staticmethod
    def define_network(config) -> "StarDistConvnextUnet":
        net = StarDistConvnextUnet(config)
        net.init_net(init_type=config.init_type, init_gain=config.init_gain)
        net.print_network()

        return net

    
######################################################################
#                   DIST LOSS
######################################################################

class DistLoss(nn.Module):
    def __init__(self, lambda_reg=0., norm_by_mask=True):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.criterion = nn.L1Loss(reduction="none")
        self.norm_by_mask = norm_by_mask

    def forward(self, input, target, mask=torch.tensor(1.), dim=1, eps=1e-9):
        actual_loss = mask * self.criterion(input, target)
        norm_mask = mask.mean() + eps if self.norm_by_mask else 1
        if self.lambda_reg > 0:
            reg_loss = (1 - mask) * torch.abs(input)

            loss = actual_loss.mean(dim=dim) / norm_mask + self.lambda_reg * reg_loss.mean(dim=dim)

        else:
            loss = actual_loss.mean(dim=dim) / norm_mask
        return loss.mean()

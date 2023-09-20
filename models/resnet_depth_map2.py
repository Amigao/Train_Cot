from matplotlib.pyplot import xcorr, xlim
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .rgb_depth_resnet import RGBDepthResnet
from .map_decoder import MapDecoder

class ResnetDepthMap(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dtype          = params.dtype
        self.device         = params.device
        self.batch_size     = params.batch_size
        self.bottleneck_dim = params.bottleneck_dim
        self.out_dim        = (params.output_size[1], params.output_size[0])
        
        self.eps = 1e-06

        # RGB+Depth encoder
        self.encoder = RGBDepthResnet(params)

        # projection
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=self.bottleneck_dim, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True))

        # z transformation layer
        self.z_transform = nn.Sequential(
            nn.Linear(self.bottleneck_dim+3, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True))

        # Fuse current and previous maps
        self.fuse_maps = nn.Sequential(
            nn.Linear(2*self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True))

        # decoder input layer
        self.dec_input = nn.Sequential(
            nn.Linear(self.bottleneck_dim, 7*7*128),
            nn.ReLU(inplace=True))

        # decoder
        self.decoder = MapDecoder(params)

    def forward(self, rgb_img, depth_img, dx, dy, dtheta, prev_z):
        # Encoder
        x = self.encoder(rgb_img, depth_img)

        # Projection
        curr_z = self.projection(x)

        T_z = torch.stack((dx, dy, dtheta), dim=1)

        # Rotate and translate previous latent vector
        prev_z = torch.cat((prev_z, T_z), dim=1)
        prev_z = self.z_transform(prev_z)

        # Update hidden state
        new_z = torch.cat((curr_z, prev_z), dim=1)
        new_z = self.fuse_maps(new_z)

        # And transform data to be used in decoder
        x = self.dec_input(new_z)
        x = x.view(-1,128,7,7)

        # Decoder
        x = self.decoder(x)

        return x, new_z

    def init_hidden_state(self):
        return torch.zeros(self.batch_size, self.bottleneck_dim).to(self.device).type(self.dtype)

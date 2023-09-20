import torch
import torch.nn as nn
import torch.nn.functional as F

#from .rgb_depth_resnet import RGBDepthResnet
from .rgb_depth_encoder import RGBDepthEncoder
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
        self.encoder = RGBDepthEncoder(params)

        # projection
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=self.bottleneck_dim, kernel_size=2, bias=False),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.Dropout(p=0.5),
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
        self.rnn = nn.LSTMCell(self.bottleneck_dim, self.bottleneck_dim)

        # decoder input layer
        self.dec_input = nn.Sequential(
            nn.Linear(self.bottleneck_dim, 7*7*64),
            nn.ReLU(inplace=True))

        # decoder
        self.decoder = MapDecoder(params)

    def forward(self, rgb_img, depth_img, dx, dy, dtheta, hc):
        # Encoder
        x = self.encoder(rgb_img, depth_img)

        # Projection
        x = self.projection(x)

        T_z = torch.stack((dx, dy, dtheta), dim=1)

        h_n = hc[0]
        c_n = hc[1]

        # Rotate and translate previous latent vector
        h_n = torch.cat((h_n, T_z), dim=1)
        h_n = self.z_transform(h_n)

        # Fuse using LSTM
        (h_n, c_n) = self.rnn(x, (h_n, c_n))

        # And transform data to be used in decoder
        x = self.dec_input(h_n)
        x = x.view(-1,64,7,7)

        # Decoder
        x = self.decoder(x)

        return x, (h_n, c_n)

    def init_hidden_state(self):
        h_n = torch.zeros(self.batch_size, self.bottleneck_dim).to(self.device).type(self.dtype)
        c_n = torch.zeros(self.batch_size, self.bottleneck_dim).to(self.device).type(self.dtype)
        return (h_n, c_n)

    def repackage_hidden(self, z):
        return tuple(item.detach() for item in z)

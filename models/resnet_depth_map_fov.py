from numpy import reshape
import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F

from .rgb_depth_resnet import RGBDepthResnet

class ResnetDepthMap(nn.Module):
    def __init__(self, params):
        super(ResnetDepthMap, self).__init__()
        # Params
        self.dtype              = params.dtype
        #self.device             = params.device
        #self.batch_size         = params.batch_size // torch.cuda.device_count()
        self.bottleneck_dim     = int(params.bottleneck_dim/2)
        self.output_channels    = params.output_channels
        self.out_dim            = (params.output_size[1], params.output_size[0])
        self.eps = 1e-06

        # Init hidden state
        # self.reset_hidden()   #torch.zeros([self.batch_size, 256, 7, 7]).to(device).type(self.dtype)

        # RGB+Depth encoder
        self.encoder = RGBDepthResnet(params)

        # projection
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, bias=False),
            nn.Flatten(),
            nn.Linear(6*11*128, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, 7*7*256),
            nn.ReLU(inplace=True))

        # decoder
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))

        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels=params.output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(params.output_channels))

    def forward(self, rgb_img, depth_img, dx, dy, dtheta, z):
        # Encoder
        x = self.encoder(rgb_img, depth_img)

        # Projection
        curr_z = self.projection(x).view(-1,256,7,7)

        # Separate map from weight map
        z_map, z_weight = curr_z.chunk(2, dim=1)

        # Decoder
        x = self.convTrans1(z_map)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = self.convTrans4(x)
        x = self.convTrans5(x)
        
        x = F.interpolate(x, size=self.out_dim)
        return x, z_weight

    # def repackage_hidden(self):
    #     self.z = self.z.detach()

    # def reset_hidden(self):
    #     print('bla reset')
    #     #self.z = torch.zeros([self.batch_size, 256, 7, 7]).cuda().type(self.dtype)
    #     self.z = None
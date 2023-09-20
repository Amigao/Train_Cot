from numpy import reshape
import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F

from .rgb_depth_resnet import RGBDepthResnet
from .map_decoder import MapDecoder

class ResnetDepthMap(nn.Module):
    def __init__(self, params):
        super(ResnetDepthMap, self).__init__()
        # Params
        self.dtype              = params.dtype
        #self.device             = params.device
        #self.batch_size         = params.batch_size // torch.cuda.device_count()
        self.bottleneck_dim     = params.bottleneck_dim
        self.output_channels    = params.output_channels
        self.out_dim            = (params.output_size[1], params.output_size[0])
        self.eps                = 1e-06 # keep 1e-06 to avoid problems when using float16
        self.max_w              = 1e3

        # Init hidden state
        # self.reset_hidden()   #torch.zeros([self.batch_size, 256, 7, 7]).to(device).type(self.dtype)

        # RGB+Depth encoder
        self.encoder = RGBDepthResnet(params)

        # bottleneck
        # self.bottleneck = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((7,7)),
        #     nn.Conv2d(in_channels=512, out_channels=2*self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=2*self.bottleneck_dim, out_channels=2*self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
        #     nn.ReLU(inplace=True))

        #pool_dim = int(np.round(np.sqrt(2*self.bottleneck_dim/512)))

        # projection
        self.projection = nn.Sequential(
            #nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, bias=False),
            #nn.Flatten(),
            #nn.Linear(6*11*128, self.bottleneck_dim),
            nn.Conv2d(in_channels=512, out_channels=self.bottleneck_dim, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            #nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, 7*7*256),
            nn.ReLU(inplace=True))

        # transform
        # self.transform = nn.Sequential(
        #     nn.Linear(3, 3),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(3, 3),
        #     nn.ReLU(inplace=True))

        # decoder
        self.decoder = MapDecoder(params)

    def forward(self, rgb_img, depth_img, dx, dy, dtheta, z):

        # if self.z is None:
        #     print('bla')
        #     self.z = torch.zeros([rgb_img.shape[0], 256, 7, 7]).to(rgb_img.device).type(self.dtype)
        # else:
        #     print('not bla')

        # Encoder
        x = self.encoder(rgb_img, depth_img)

        # Projection
        curr_z = self.projection(x).view(-1,256,7,7)

        # transf = torch.stack((dx, dy, dtheta), dim=1)
        # transf = self.transform(transf)
        # dx2, dy2, dtheta2 = transf[:,0], transf[:,1], transf[:,2]

        # Apply transformation to the latent vector z
        prev_z = kornia.geometry.transform.rotate(z, -dtheta*180.0/torch.pi)
        translation = torch.stack((dy/0.05/(200.0/7.0), dx/0.05/(200.0/7.0)), dim=1)
        prev_z = kornia.geometry.transform.translate(prev_z, translation)

        # Separate map from weight map
        z_map, z_weight = curr_z.chunk(2, dim=1)
        prev_z_map, prev_z_weight = prev_z.chunk(2, dim=1)

        # Constraint current z weight between [0,1]
        z_weight = torch.sigmoid(z_weight)

        # And fuse then
        new_z_map = (z_map*z_weight + prev_z_map*prev_z_weight) / (z_weight + prev_z_weight + self.eps)
        # Sum weights and constraint new weight to max_w to avoid numerical issues
        new_z_weight = torch.clamp(z_weight + prev_z_weight, max=self.max_w)   #/2
        new_z = torch.cat((new_z_map, new_z_weight), dim=1)

        # Decoder
        x = self.decoder(new_z_map)
        
        return x, new_z

    # def repackage_hidden(self):
    #     self.z = self.z.detach()

    # def reset_hidden(self):
    #     print('bla reset')
    #     #self.z = torch.zeros([self.batch_size, 256, 7, 7]).cuda().type(self.dtype)
    #     self.z = None
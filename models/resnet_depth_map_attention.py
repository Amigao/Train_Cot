from numpy import reshape
import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F

from .rgb_depth_resnet import RGBDepthResnet
#from .rgb_depth_encoder import RGBDepthEncoder
#from .map_decoder import MapDecoder
from .resnet_decoder import ResNet18Dec
from .non_local import NLBlockND

class ResnetDepthMap(nn.Module):
    def __init__(self, params):
        super(ResnetDepthMap, self).__init__()
        # Params
        self.dtype              = params.dtype
        self.device             = params.device
        self.batch_size         = params.batch_size
        self.bottleneck_dim     = params.bottleneck_dim
        self.output_channels    = params.output_channels
        self.out_dim            = (params.output_size[1], params.output_size[0])
        self.eps                = 1e-06 # keep 1e-06 to avoid problems when using float16

        # RGB+Depth encoder
        self.encoder = RGBDepthResnet(params)
        #self.encoder = RGBDepthEncoder(params)

        # Non-local attention block
        self.attention = NLBlockND(in_channels=512, dimension=3)

        # 3D convolutional block for temporal fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(2, 1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True))

        # decoder
        #self.decoder = MapDecoder(params)
        self.decoder = ResNet18Dec(nc=1)

    def forward(self, rgb_img, depth_img, dx, dy, dtheta, z):
        # Encoder
        x = self.encoder(rgb_img, depth_img)

        # Apply transformation to the latent vector z correspondent to previous map
        prev_z = kornia.geometry.transform.rotate(z, -dtheta*180.0/torch.pi)
        translation = torch.stack((dy/0.05/(200.0/7.0), dx/0.05/(200.0/7.0)), dim=1)
        prev_z = kornia.geometry.transform.translate(prev_z, translation)

        # Concatenate in a new dim correspondent to time
        x = torch.stack((x, prev_z), dim=2)

        # Use non-local self-attention block
        x = self.attention(x)

        # And fuse the temporal relation to current time
        new_z = self.fusion(x).view(-1,512,7,7)

        # Decoder
        x = self.decoder(new_z)

        x = F.interpolate(x, size=self.out_dim)
        
        return x, new_z

    def init_hidden_state(self):
        z = torch.zeros([self.batch_size, 512, 7, 7]).to(self.device).type(self.dtype)
        return z

    def repackage_hidden(self, z):
        return z.detach()
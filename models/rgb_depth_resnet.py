import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from .non_local import NLBlockND

class RGBDepthResnet(nn.Module):
    def __init__(self, params, out_channels):
        super().__init__()
        model = models.resnet34(pretrained=params.pretrained)

        # RGB encoder
        self.block1 = nn.Sequential(*(list(model.children())[:3]))
        self.block2 = nn.Sequential(model.maxpool, model.layer1)
        self.block3 = model.layer2
        self.block4 = model.layer3
        self.block5 = model.layer4

        # Depth encoder
        self.block1_depth = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.block2_depth = copy.deepcopy(self.block2)
        self.block3_depth = copy.deepcopy(self.block3)
        self.block4_depth = copy.deepcopy(self.block4)
        self.block5_depth = copy.deepcopy(self.block5)

        # Mixture encoder
        self.block2_mix = copy.deepcopy(self.block2)
        self.block3_mix = copy.deepcopy(self.block3)
        self.block4_mix = copy.deepcopy(self.block4)
        self.block5_mix = copy.deepcopy(self.block5)

        # Non-local attention block
        self.attention = NLBlockND(in_channels=256, dimension=3)

        # 3D convolutional block for modality fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True))

        self.conv_f = nn.Conv2d(256, out_channels, kernel_size=1, padding=0)

    def forward(self, rgb_img, depth_img):
        # Encoder block 1
        x_rgb = self.block1(rgb_img)
        x_depth = self.block1_depth(depth_img)
        x_mix = x_rgb + x_depth
        # Encoder block 2
        x_rgb = self.block2(x_rgb)
        x_depth = self.block2_depth(x_depth)
        x_mix = self.block2_depth(x_mix)
        x_mix = x_rgb + x_depth + x_mix
        # Encoder block 3
        x_rgb = self.block3(x_rgb)
        x_depth = self.block3_depth(x_depth)
        x_mix = self.block3_depth(x_mix)
        x_mix = x_rgb + x_depth + x_mix
        # Encoder block 4
        x_rgb = self.block4(x_rgb)
        x_depth = self.block4_depth(x_depth)
        x_mix = self.block4_depth(x_mix)
        x_mix = x_rgb + x_depth + x_mix
        # Encoder block 5
        # x_rgb = self.block5(x_rgb)
        # x_depth = self.block5_depth(x_depth)
        # x_mix = self.block5_depth(x_mix)

        x = torch.stack((x_rgb, x_depth, x_mix), dim=2)

        # Use attention block
        x = self.attention(x)
        # And 3D conv for fusion
        x = self.fusion(x).squeeze(2)

        x = self.conv_f(x)

        # Adjust shape with adaptive average pool
        #x = F.adaptive_avg_pool2d(x, (7,7))

        return x

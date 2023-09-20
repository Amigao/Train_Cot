import copy
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class RGBDepthEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        model = models.resnet18(pretrained=params.pretrained)

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

    def forward(self, rgb_img, depth_img):
        # Encoder
        x = self.block1(rgb_img)
        x_depth = self.block1_depth(depth_img)
        #x = x + x_depth
        x = self.block2(x)
        x_depth = self.block2_depth(x_depth)
        #x = x + x_depth
        x = self.block3(x)
        x_depth = self.block3_depth(x_depth)
        #x = x + x_depth
        x = self.block4(x)
        x_depth = self.block4_depth(x_depth)
        #x = x + x_depth
        x = self.block5(x)
        x_depth = self.block5_depth(x_depth)
        #x = x + x_depth

        x = torch.cat((x, x_depth), dim=1)

        return x

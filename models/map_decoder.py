import torch
import torch.nn as nn
import torch.nn.functional as F

class MapDecoder(nn.Module):
    def __init__(self, params):
        super(MapDecoder, self).__init__()
        # Params
        self.output_channels    = params.output_channels
        self.out_dim            = (params.output_size[1], params.output_size[0])

        # decoder
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False)),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False)),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False)),
            #nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))

        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            #nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False)),
            #nn.BatchNorm2d(4),
            nn.ReLU(inplace=True))

        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            #nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(32, out_channels=params.output_channels, kernel_size=3, padding=1, bias=False)))

    def forward(self, input):
        # Decoder
        x = self.convTrans1(input)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = self.convTrans4(x)
        x = self.convTrans5(x)
        return x
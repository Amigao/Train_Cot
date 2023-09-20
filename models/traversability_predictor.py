import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_depth_map3 import ResnetDepthMap

class TravPredictor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dtype          = params.dtype
        self.device         = params.device
        self.batch_size     = params.batch_size
        self.bottleneck_dim = params.bottleneck_dim
        self.out_dim        = (params.output_size[1], params.output_size[0])

        # mapper
        self.mapper = ResnetDepthMap(params)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.bottleneck_dim+5, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid())

    def forward(self, rgb_img, depth_img, dx, dy, dtheta, x_seq, y_seq, theta_seq, v_seq, omega_seq, hc):
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

        x = torch.zeros_like(x_seq)

        # Run the predictor for each value in the sequence
        for t in x_seq.shape[1]:
            in_state = torch.cat((h_n, x_seq[:,t:t+1], y_seq[:,t:t+1], theta_seq[:,t:t+1], v_seq[:,t:t+1], omega_seq[:,t:t+1]), dim=1)
            x[:,t] = self.predictor(in_state)

        return x, (h_n, c_n)

    def init_hidden_state(self):
        h_n = torch.zeros(self.batch_size, self.bottleneck_dim).to(self.device).type(self.dtype)
        c_n = torch.zeros(self.batch_size, self.bottleneck_dim).to(self.device).type(self.dtype)
        return (h_n, c_n)

    def repackage_hidden(self, z):
        return tuple(item.detach() for item in z)

"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import kornia
import numpy as np
from torch import nn
from efficientnet_pytorch import EfficientNet
from .rgb_depth_resnet import RGBDepthResnet
from torchvision.models.resnet import resnet18

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        #self.up = nn.Upsample(size=(11, 21), mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.downsample = downsample

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        if self.downsample == 16:
            in_channels = 320+112
            out_channels = 512
        elif self.downsample == 8:
            in_channels = 112 + 40
            out_channels = 128

        self.up1 = Up(in_channels, out_channels)
        self.depthnet = nn.Conv2d(out_channels, self.D + self.C, kernel_size=1, padding=0)

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

            if self.downsample == 8 and idx == 10:
                break

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x

        if self.downsample == 16:
            x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        elif self.downsample == 8:
            x = self.up1(endpoints['reduction_4'], endpoints['reduction_3'])

        return x

    def forward(self, x):
        x = self.get_eff_depth(x)
        
        # Depth
        x = self.depthnet(x)

        depth = x[:, :self.D].softmax(dim=1)
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return new_x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)

        self.conv1 = nn.Conv2d(inC//2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x

class LiftSplatShoot(nn.Module):
    def __init__(self, params):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf  = params.grid_conf
        self.input_size = (params.input_size[1], params.input_size[0])
        self.batch_size = params.batch_size
        self.dtype      = params.dtype
        self.eps        = 1e-6

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'])

        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16#8 
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        #self.camencode = CamEncode(self.D, 2*self.camC, self.downsample)
        self.cam_encode = RGBDepthResnet(params, self.D + self.camC)
        self.bev_encode = BevEncode(inC=self.camC, outC=params.output_channels)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.input_size
        fH, fW = int(np.ceil(ogfH / self.downsample)), int(np.ceil(ogfW / self.downsample))
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    '''
    This get_geometry comes from the Fiery paper
    and I believe it's a little more organized
    '''
    def get_geometry(self, intrinsics, extrinsics):
        """
        Calculate the (x, y, z) 3D position of the features.
        - Intrinsics is a 3x3 matrix for projection
        - Extrinsics is a 4x4 matrix with rotation and translation to a reference frame (center of the robot)
        """
        # separate rotation and translation from extrinsics matrix
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        # get batch and number of cams dimensions
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats  = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats  = geom_feats.view(Nprime, 3)
        batch_ix    = torch.cat([torch.full([Nprime//B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats  = torch.cat((geom_feats, batch_ix), 1)

        # Mask out points that are outside the considered spatial extent.
        mask = ((geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) &
                (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) &
                (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2]))
        x = x[mask]
        geom_feats = geom_feats[mask]

        # Sort tensors so that those within the same voxel are consecutives.
        ranks = (geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
                + geom_feats[:, 1] * (self.nx[2] * B)
                + geom_feats[:, 2] * B
                + geom_feats[:, 3])
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    #def get_voxels(self, color_img, depth_img, rots, trans, intrins, post_rots, post_trans):
    def get_voxels(self, color_img, depth_img, intrinsics, extrinsics):
        #geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        geom = self.get_geometry(intrinsics, extrinsics)

        '''
        Return B x N x D x H/downsample x W/downsample x C
        '''
        B, N, C, imH, imW = color_img.shape

        color_img = color_img.view(B*N, C, imH, imW)
        depth_img = depth_img.view(B*N, 2, imH, imW)

        x = self.cam_encode(color_img, depth_img)

        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.camC)].unsqueeze(2)

        x = x.view(B, N, self.camC, self.D, int(np.ceil(imH/self.downsample)), int(np.ceil(imW/self.downsample)))
        x = x.permute(0, 1, 3, 4, 5, 2)

        x = self.voxel_pooling(geom, x)

        # Now, x is a grid of shape (B x C x X x Y)

        return x

    #def forward(self, color_img, depth_img, rots, trans, intrins, post_rots, post_trans, prev_z, dx, dy, dtheta):
    def forward(self, color_img, depth_img, intrinsics, extrinsics, dx, dy, dtheta, z):
        #curr_z = self.get_voxels(color_img, depth_img, rots, trans, intrins, post_rots, post_trans)
        curr_z = self.get_voxels(color_img, depth_img, intrinsics, extrinsics)

        # Rotate the grid 180 deg so x points top and y is left
        curr_z = curr_z.flip(2).flip(3)

        # Apply transformation to the latent vector z
        prev_z = kornia.geometry.transform.rotate(z, -dtheta*180.0/torch.pi)
        translation = torch.stack((dy/0.05, dx/0.05), dim=1)
        prev_z = kornia.geometry.transform.translate(prev_z, translation)

        # Separate map from weight map
        curr_map, curr_conf = curr_z.chunk(2, dim=1)
        prev_map, prev_conf = prev_z.chunk(2, dim=1)

        # Constraint current z weight between [0,1]
        curr_conf = torch.sigmoid(curr_conf)

        new_map = (curr_map*curr_conf + prev_map*prev_conf)/(curr_conf + prev_conf + self.eps)
        new_conf = curr_conf + prev_conf

        new_z = torch.cat((new_map, new_conf), dim=1)

        x = self.bev_encode(new_map)
        return x, new_z

    def repackage_hidden(self, z):
        z = z.detach()
        return z

    def init_hidden_state(self):
        z = torch.zeros([self.batch_size, self.camC, 200, 200]).cuda().type(self.dtype)
        return z
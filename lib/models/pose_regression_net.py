# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.v2v_net import V2VNet
from models.project_layer import ProjectLayer
from utils.cameras import project_pose
from utils.transforms import get_affine_transform as get_transform
from utils.transforms import affine_transform_pts_cuda as do_transform



class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x


class DenormProject(nn.Module):
    def __init__(self, cfg):
        super(DenormProject, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.ori_image_size = cfg.NETWORK.ORI_IMAGE_SIZE

    def forward(self, meta, nerf_model, grids, feats2d=None):
        nview = len(meta)
        denorm_imgs = [metan['denorm_img'].permute(0, 3, 1, 2) for metan in meta]  # [tensor(bs, rgb,h ,w)...]
        batch_size = denorm_imgs[0].shape[0]
        resize_transform = meta[0]['transform'][0]
        w, h = self.image_size
        device = grids.device
        nbins = self.cube_size[0] * self.cube_size[1] * self.cube_size[2]
        # 投影 到  grid中
        cubes = torch.zeros(batch_size, denorm_imgs[0].shape[1], 1, nbins, nview, device=device)
        if feats2d is not None:
            cubes_feats = torch.zeros(batch_size, feats2d[0].shape[1], 1, nbins, nview, device=device)
        for i in range(batch_size):
            for c in range(nview):
                trans = torch.as_tensor(
                    resize_transform,
                    dtype=torch.float,
                    device=device)
                cam = {}
                for k, v in meta[c]['camera'].items():
                    cam[k] = v[i]
                xy = project_pose(grids[i], cam)
                xy = torch.clamp(xy, -1.0, max(self.ori_image_size[0], self.ori_image_size[1]))
                xy = do_transform(xy, trans)
                # xy = xy * torch.tensor(
                #     [w, h], dtype=torch.float, device=device) / torch.tensor(
                #     self.img_size, dtype=torch.float, device=device)
                sample_grid = xy / torch.tensor(
                    [w - 1, h - 1], dtype=torch.float,
                    device=device) * 2.0 - 1.0
                sample_grid = sample_grid.view(1, 1, nbins, 2)
                #  这个是denorm img
                cubes[i:i + 1, :, :, :, c] += F.grid_sample(denorm_imgs[c][i:i + 1, :, :, :], sample_grid,
                                                            align_corners=True)
                if feats2d is not None:
                    cubes_feats[i:i + 1, :, :, :, c] += F.grid_sample(feats2d[c][i:i + 1, :, :, :], sample_grid,
                                                                align_corners=True)

        if feats2d is not None:
            cubes = torch.cat([cubes, cubes_feats], dim=1)
            del cubes_feats

        # mean
        mean = torch.sum(cubes / nview, dim=-1, keepdim=True)
        # var


        # var
        var = torch.sum((cubes - mean) ** 2, dim=-1, keepdim=True)
        var = var / nview
        var = torch.exp(-var)
        global_volume = torch.cat([mean, var], dim=1) # [bs, 6, ...]
        del cubes, mean, var


        volumes = []
        for i in range(batch_size):
            g_volume = global_volume[i].view(-1, nbins).permute(1, 0).contiguous()
            points = grids[i].view(-1, 3).contiguous()
            density = nerf_model.query_density(points, g_volume)
            alpha = 1 - torch.exp(-density)
            v = alpha.view(1, self.cube_size[0], self.cube_size[1], self.cube_size[2])
            volumes.append(v)
        x = torch.stack(volumes, dim=0)
        return x


class PoseRegressionNet(nn.Module):
    def __init__(self, cfg):
        super(PoseRegressionNet, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE

        self.project_layer = ProjectLayer(cfg)
        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, cfg.NETWORK.NUM_JOINTS)
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

        self.denorm_project = DenormProject(cfg)

    def forward(self, all_heatmaps, meta, grid_centers, nerf_model, feats2d=None):
        # grid_centers：所有bs ，第n个人的 中心位置012，flag3， topk_value,4  grid_centers[:, n]---(bs, 5)
        batch_size = all_heatmaps[0].shape[0]
        num_joints = all_heatmaps[0].shape[1]
        device = all_heatmaps[0].device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)
        cubes, grids = self.project_layer(all_heatmaps, meta, self.grid_size, grid_centers, self.cube_size)
        # '''这边得到的cubes是包含bs，每个人得到3D 网格；grids保存 在世界坐标系的3D坐标'''
        # '''inidividual -> cubes：整个大网格， offset：在2D平面的坐标 '''
        # '''先得到中心点坐标，然后在fine grid上相应空间采样，其余mask，返回grid和cubes'''
        # nerf
        alpha = self.denorm_project(meta, nerf_model, grids, feats2d)
        cubes = cubes * (0.5*alpha + 0.5)

        index = grid_centers[:, 3] >= 0
        valid_cubes = self.v2v_net(cubes[index])
        pred[index] = self.soft_argmax_layer(valid_cubes, grids[index])

        return pred # 每个bs，第n号人，每个关节的3D坐标   pred[:, n, :, 0:3] --- (bs, 15, 3)

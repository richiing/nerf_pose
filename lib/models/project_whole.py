# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cameras import project_pose
from utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.ori_image_size = cfg.NETWORK.ORI_IMAGE_SIZE

        self.space_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.space_center = cfg.MULTI_PERSON.SPACE_CENTER
        self.voxels_per_axis = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE


        self.grid = self.compute_grid(self.space_size, self.space_center, self.voxels_per_axis)
        self.sample_grid = {}
        self.sample_grid_img = {}

    def compute_grid(self, boxSize, boxCenter, nBins):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0])
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1])
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2])
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2]
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def project_grid(self, camera, w, h, nbins, resize_transform, device):
        grid = self.grid.to(device)
        xy = project_pose(grid, camera)
        xy = torch.clamp(xy, -1.0, max(self.ori_image_size[0], self.ori_image_size[1]))
        resize_transform = torch.as_tensor(resize_transform,
                        dtype=torch.float,
                        device=device)
        xy = do_transform(xy, resize_transform)
        xy = xy * torch.tensor(
            [w, h], dtype=torch.float, device=device) / torch.tensor(
            self.image_size, dtype=torch.float, device=device)
        sample_grid = xy / torch.tensor(
            [w - 1, h - 1], dtype=torch.float,
            device=device) * 2.0 - 1.0
        sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)
        return sample_grid

    def forward(self, heatmaps, meta):
        resize_transform = meta[0]['transform'][0]
        cameras = [metan['camera'] for metan in meta]  # [{cam0*bs}, *4]
        heatmaps = torch.stack(heatmaps, dim=1) # [bs, n, joint, h, w]
        device = heatmaps.device
        batch_size = heatmaps.shape[0]
        n = heatmaps.shape[1]
        num_joints = heatmaps.shape[2]
        nbins = self.voxels_per_axis[0] * self.voxels_per_axis[1] * self.voxels_per_axis[2]
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, device=device)
        w, h = self.heatmap_size
        w_img, h_img = self.image_size


        for i in range(batch_size):
            curr_seq = meta[0]['seq'][i]
            if curr_seq not in self.sample_grid:
                print("=> HDN feats2D ", curr_seq)
                sample_grids = torch.zeros(n, 1, nbins, 2, device=device)
                for c in range(n):
                    cam = {}
                    for k, v in cameras[c].items():
                        cam[k] = v[i]
                    sample_grids[c] = self.project_grid(cam, w, h, nbins, resize_transform,
                                                        device).squeeze(0)
                self.sample_grid[curr_seq] = sample_grids

            if curr_seq not in self.sample_grid_img:
                print("=>  HDN denorm_imgs ", curr_seq)
                sample_grids = torch.zeros(n, 1, nbins, 2, device=device)
                for c in range(n):
                    cam = {}
                    for k, v in cameras[c].items():
                        cam[k] = v[i]
                    sample_grids[c] = self.project_grid(cam, w_img, h_img, nbins, resize_transform,
                                                        device).squeeze(0)
                self.sample_grid_img[curr_seq] = sample_grids

            shared_sample_grid = self.sample_grid[curr_seq].to(device)
            cubes[i] = torch.mean(F.grid_sample(heatmaps[i], shared_sample_grid, align_corners=True), dim=0).squeeze(0)


        cubes = cubes.clamp(0.0, 1.0)
        cubes = cubes.view(batch_size, num_joints, self.voxels_per_axis[0], self.voxels_per_axis[1],
                           self.voxels_per_axis[2])
        return cubes
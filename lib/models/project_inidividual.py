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

        # constants for back-projection
        self.whole_space_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)
        self.whole_space_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.ind_space_size = torch.tensor(cfg.PICT_STRUCT.GRID_SIZE)
        self.voxels_per_axis = torch.tensor(cfg.PICT_STRUCT.CUBE_SIZE, dtype=torch.int32)
        self.fine_voxels_per_axis = (self.whole_space_size / self.ind_space_size * (self.voxels_per_axis - 1)).int() + 1

        # fine_grid 中，每mm有多少个体素块
        self.scale = (self.fine_voxels_per_axis.float() - 1) / self.whole_space_size
        # fine_grid 中，
        # - 个人grid 中心相对于 个人grid左下角的体素块index
        # - scale * 左下角的3d坐标 = - 相对于原点 左下角所代表的体素块index。。
        self.bias = - self.ind_space_size / 2.0 / self.whole_space_size * (self.fine_voxels_per_axis - 1) \
                    - self.scale * (self.whole_space_center - self.whole_space_size / 2.0)

        self.save_grid()
        self.sample_grid = {}

    def save_grid(self):
        print("=> save the 3D grid for feature sampling")
        # grid = self.compute_grid(self.ind_space_size, self.whole_space_center, self.voxels_per_axis)
        # grid = grid.view(self.voxels_per_axis[0], self.voxels_per_axis[1], self.voxels_per_axis[2], 3)
        # self.center_grid = torch.stack([grid[:, :, 0, :2].reshape(-1, 2), grid[:, 0, :, ::2].reshape(-1, 2), \
        #                                 grid[0, :, :, 1:].reshape(-1, 2)])
        self.fine_grid = self.compute_grid(self.whole_space_size, self.whole_space_center, self.fine_voxels_per_axis)
        return

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
            grid1Dz + boxCenter[2],
            indexing='ij'
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def project_grid(self, camera, w, h, nbins, resize_transform, device):
        xy = project_pose(self.fine_grid.to(device), camera)
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

    '''
    only compute the projected 2D finer grid once for each sequence
    '''

    def compute_sample_grid(self, heatmaps, i, voxels_per_axis, seq, cameras, resize_transform):
        device = heatmaps.device
        nbins = voxels_per_axis[0] * voxels_per_axis[1] * voxels_per_axis[2]
        n = heatmaps.shape[1]
        w, h = self.heatmap_size

        # compute the sample grid
        sample_grids = torch.zeros(n, 1, 1, nbins, 2, device=device)
        for c in range(n):
            cam = {}
            for k, v in cameras[c].items():
                cam[k] = v[i]
            sample_grid = self.project_grid(cam, w, h, nbins, resize_transform, device)
            sample_grids[c] = sample_grid
        self.sample_grid[seq] = sample_grids.view(n, voxels_per_axis[0], voxels_per_axis[1], voxels_per_axis[2], 2)

    def forward(self, heatmaps, meta, proposal_centers):
        '''heatmaps: [tensor(bs,15,h,w)*4]
        meta: [{},*4]
        proposal_centers: tensor(bs, 5)---3D index + invalid flag + value'''
        heatmaps = torch.stack(heatmaps, dim=1) # (bs ,nview, 15, h, w)
        resize_transform = meta[0]['transform'][0]
        cameras = [metan['camera'] for metan in meta]  # [{cam0*bs}, *4]
        device = heatmaps.device
        n = heatmaps.shape[1]
        num_joints = heatmaps.shape[2]
        batch_size = heatmaps.shape[0]
        voxels_per_axis = self.voxels_per_axis.to(device)
        nbins = voxels_per_axis[0] * voxels_per_axis[1] * voxels_per_axis[2]

        cubes = torch.zeros(batch_size, num_joints, voxels_per_axis[0], voxels_per_axis[1], voxels_per_axis[2], device=device)
        grids = torch.zeros(batch_size, nbins, 3, device=device)




        for index in range(batch_size):
            curr_seq = meta[0]['seq'][index]
            if curr_seq not in self.sample_grid:
                print("=> JLN ", curr_seq)
                self.compute_sample_grid(heatmaps, index, self.fine_voxels_per_axis.to(device), curr_seq, cameras,
                                         resize_transform)
            if proposal_centers[index][3] >= 0:
                sample_grid = self.sample_grid[curr_seq].to(device)
                # compute the index of the top left point in the fine-grained volume
                # proposal centers: [batch_size, 5].
                centers_tl = torch.round(proposal_centers[:, :3].float() * self.scale + self.bias).int()

                # mask the feature volume outside the bounding box


                '''在中心3D，得到边界框的3D点，然后取相应的体素块，设置成新的体素cube，grid也就是对应的位置'''









        # compute the index of the top left point in the fine-grained volume
        # proposal centers: [batch_size, 7]
        centers_tl = torch.round(proposal_centers[:, :3].float() * self.scale + self.bias).int()
        offset = centers_tl.float() / (
                    self.fine_voxels_per_axis - 1) * self.whole_space_size - self.whole_space_size / 2.0 + self.ind_space_size / 2.0

        # mask the feature volume outside the bounding box
        mask = ((1 - proposal_centers[:, 5:7]) / 2 * (self.voxels_per_axis[0:2] - 1)).int()
        mask[mask < 0] = 0
        # the vertical length of the bounding box is kept fixed as 2000mm
        mask = torch.cat([mask, torch.zeros((num_people, 1), device=device, dtype=torch.int32)], dim=1)

        # compute the valid range to filter the outsider
        start = torch.where(centers_tl + mask >= 0, centers_tl + mask, torch.zeros_like(centers_tl))
        end = torch.where(centers_tl + self.voxels_per_axis - mask <= self.fine_voxels_per_axis,
                          centers_tl + self.voxels_per_axis - mask, self.fine_voxels_per_axis)

        # construct the feature volume
        for i in range(num_people):
            if torch.sum(start[i] >= end[i]) > 0:
                continue
            sample_grid = self.sample_grid[curr_seq]
            sample_grid = sample_grid[:, start[i, 0]:end[i, 0], start[i, 1]:end[i, 1], start[i, 2]:end[i, 2]].reshape(n,
                                                                                                                      1,
                                                                                                                      -1,
                                                                                                                      2)

            accu_cubes = torch.mean(F.grid_sample(heatmaps[index], sample_grid, align_corners=True), dim=0).view(
                num_joints, end[i, 0] - start[i, 0], end[i, 1] - start[i, 1], end[i, 2] - start[i, 2])
            cubes[i, :, start[i, 0] - centers_tl[i, 0]:end[i, 0] - centers_tl[i, 0],
            start[i, 1] - centers_tl[i, 1]:end[i, 1] - centers_tl[i, 1],
            start[i, 2] - centers_tl[i, 2]:end[i, 2] - centers_tl[i, 2]] = accu_cubes
            del sample_grid

        cubes = cubes.clamp(0.0, 1.0)
        del mask
        return cubes, offset, start, end, centers_tl
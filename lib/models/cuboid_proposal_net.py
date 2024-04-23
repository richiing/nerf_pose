# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.v2v_net import V2VNet
from models.project_whole import ProjectLayer
from core.proposal import nms


class ProposalLayer(nn.Module):
    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.cube_size = torch.tensor(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.threshold = cfg.MULTI_PERSON.THRESHOLD

    def filter_proposal(self, topk_index, gt_3d, num_person):
        batch_size = topk_index.shape[0]
        cand_num = topk_index.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = topk_index[i].reshape(cand_num, 1, -1)
            gt = gt_3d[i, :num_person[i]].reshape(1, num_person[i], -1)

            dist = torch.sqrt(torch.sum((cand - gt)**2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > 500.0] = -1.0

        return cand2gt

    def get_real_loc(self, index):
        device = index.device
        cube_size = self.cube_size.to(device=device, dtype=torch.float)
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = index.float() / (cube_size - 1) * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, root_cubes, meta):
        batch_size = root_cubes.shape[0]

        topk_values, topk_unravel_index = nms(root_cubes.detach(), self.num_cand)
        topk_unravel_index = self.get_real_loc(topk_unravel_index)

        grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=root_cubes.device)
        grid_centers[:, :, 0:3] = topk_unravel_index
        grid_centers[:, :, 4] = topk_values

        # match gt to filter those invalid proposals for training/validate PRN
        if self.training and ('roots_3d' in meta[0] and 'num_person' in meta[0]):
            gt_3d = meta[0]['roots_3d'].float()
            num_person = meta[0]['num_person']
            cand2gt = self.filter_proposal(topk_unravel_index, gt_3d, num_person)
            grid_centers[:, :, 3] = cand2gt
        else:
            grid_centers[:, :, 3] = (topk_values > self.threshold).float() - 1.0  # if ground-truths are not available.



        return grid_centers


class DenormProject(nn.Module):
    def __init__(self, cfg):
        super(DenormProject, self).__init__()
        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER
    def forward(self, meta, nerf_model, grids, sample_grid_img, device, initial_cubes, feats2d=None):
        grids = grids.to(device)

        nview = len(meta)
        denorm_imgs = [metan['denorm_img'].permute(0, 3, 1, 2) for metan in meta] # [tensor(bs, rgb,h ,w)...]
        batch_size = denorm_imgs[0].shape[0]
        nbins = self.cube_size[0] * self.cube_size[1] * self.cube_size[2]
        # 投影 到  grid中
        cubes = torch.zeros(batch_size, denorm_imgs[0].shape[1], 1, nbins, nview, device=device)
        if feats2d is not None:
            cubes_feats = torch.zeros(batch_size, feats2d[0].shape[1], 1, nbins, nview, device=device)

        for i in range(batch_size):
            curr_seq = meta[0]['seq'][i]
            shared_sample_grid_img = sample_grid_img[curr_seq].to(device)

            for c in range(nview):
                sample_grid_get = shared_sample_grid_img[c].view(1, 1, nbins, 2)
                cubes[i:i + 1, :, :, :, c] += F.grid_sample(denorm_imgs[c][i:i + 1, :, :, :], sample_grid_get,
                                                                  align_corners=True)
                if feats2d is not None:
                    sample_grid_get = shared_sample_grid_img[c].view(1, 1, nbins, 2)# [tensor(bs, n, 1, nbins, 2)]->1,1,nbins,2
                    cubes_feats[i:i + 1, :, :, :, c] += F.grid_sample(feats2d[c][i:i + 1, :, :, :], sample_grid_get,
                                                                align_corners=True)
        if feats2d is not None:
            cubes = torch.cat([cubes, cubes_feats], dim=1)
            del cubes_feats
        # mean
        mean = torch.sum(cubes / nview, dim=-1, keepdim=True)
        # var
        var = torch.sum((cubes - mean) ** 2, dim=-1, keepdim=True)
        var = var / nview
        var = torch.exp(-var)
        global_volume = torch.cat([mean, var], dim=1) # [bs, 6, ...]
        del cubes, mean, var

        volumes = []
        for i in range(batch_size):
            g_volume = global_volume[i].view(-1, nbins).permute(1, 0).contiguous()
            points = grids.view(-1, 3).contiguous()
            density = nerf_model.query_density(points, g_volume)
            alpha = 1 - torch.exp(-density)

            v = alpha.view(1, self.cube_size[0], self.cube_size[1], self.cube_size[2])
            volumes.append(v)
        x = torch.stack(volumes, dim=0)#[2,1,48,48,12]
        return x

class CuboidProposalNet(nn.Module):
    def __init__(self, cfg):
        super(CuboidProposalNet, self).__init__()
        self.project_layer = ProjectLayer(cfg)
        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, 1)
        self.proposal_layer = ProposalLayer(cfg)

        self.denorm_project = DenormProject(cfg)

    def forward(self, all_heatmaps, meta, nerf_model, feats2d=None):

        initial_cubes = self.project_layer(all_heatmaps, meta)
        alpha = self.denorm_project(meta, nerf_model,
                                    self.project_layer.grid,
                                    self.project_layer.sample_grid_img,
                                    all_heatmaps[0].device,
                                    initial_cubes, feats2d)


        initial_cubes = initial_cubes * (0.5*alpha + 0.5)
        root_cubes = self.v2v_net(initial_cubes)
        root_cubes = root_cubes.squeeze(1)
        grid_centers = self.proposal_layer(root_cubes, meta)

        return root_cubes, grid_centers
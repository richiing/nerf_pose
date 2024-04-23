# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from models import pose_resnet
from models.cuboid_proposal_net import CuboidProposalNet
from models.pose_regression_net import PoseRegressionNet
from core.loss import PerJointMSELoss
from core.loss import PerJointL1Loss
from core.loss import NeRFLoss
from models.nerf_mlp import VanillaNeRFRadianceField
from models.nerf_method import NerfMethodNet

logger = logging.getLogger(__name__)

class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.backbone = backbone
        self.root_net = CuboidProposalNet(cfg)
        self.pose_net = PoseRegressionNet(cfg)

        self.USE_GT = cfg.NETWORK.USE_GT
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.model = VanillaNeRFRadianceField(
            net_depth=4,  # The depth of the MLP.
            net_width=256,  # The width of the MLP.
            skip_layer=3,  # The layer to add skip layers to.
            feature_dim=6 + cfg.NERF.DIM * 2,  # + RGB original img #self.num_joints
            net_depth_condition=1,  # The depth of the second part of MLP.
            net_width_condition=128
        )
        self.nerf_method = NerfMethodNet(cfg)
        self.use_feat_level = cfg.NERF.USE_FEAT_LEVEL
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.ori_image_size = cfg.NETWORK.ORI_IMAGE_SIZE
        self.mapping = nn.Linear(256, cfg.NERF.DIM)
        self.final_layer = nn.Conv2d(
            in_channels=cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.NERF.DIM,
            kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0
        )
        self.begin = cfg.NERF.FEAT_BEGIN_LAYER
        self.end = cfg.NERF.FEAT_END_LAYER


    def deal_with_feats2d(self, feats2d):
        batch_size = feats2d[0][0].shape[0]
        nview = len(feats2d)
        lay = len(feats2d[0])
        assert self.end <= lay, 'end is larger than lay'
        w, h = self.image_size[0], self.image_size[1]
        feats = []
        for index in range(self.begin, self.end):
            feature_index = [f[index] for f in feats2d] # [f0index, f1index, f2index, f3index]
            fff = []
            for feat in feature_index: # [tensor(bs, 256, h,w )  *4]
                ff = F.adaptive_avg_pool2d(feat, (h, w)) # tensor(bs ,256, H,W)
                # ff = ff.permute(0, 2, 3, 1).contiguous()# tensor(bs , H,W,256)
                # ff = self.mapping(ff)# tensor(bs , H,W,16)
                # ff = ff.permute(0, 3, 1, 2).contiguous()# tensor(bs , 16,H,W)
                #
                ff = self.final_conv(ff)
                fff.append(ff) #[ tensor(bs ,16, H ,W) *4]
            feats.append(fff)#[   []* 1      ]
        return feats


    def forward(self, views=None, meta=None, targets_2d=None, weights_2d=None, targets_3d=None, input_heatmaps=None, ray_batches=None):
        if views is not None:
            all_heatmaps, feats2d = [], []
            for view in views:
                heatmaps, feat2d = self.backbone(view, self.use_feat_level)
                all_heatmaps.append(heatmaps)
                feats2d.append(feat2d)
        else:
            all_heatmaps = input_heatmaps


        # all_heatmaps = targets_2d
        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]
        nview = len(all_heatmaps)
        w, h = self.image_size[0], self.image_size[1]

        # fs = self.deal_with_feats2d(feats2d)
        # feats2d = fs[0]

        feats2d = all_heatmaps # [(bs ,15,h,w) *4]  [(bs ,256,h,w) *4] ->16
        feats2d = [F.adaptive_avg_pool2d(feat, (h, w)) for feat in feats2d]


        # print("fo", f0.shape) torch.Size([1, 4, 256, 32, 60])
        # print("f1", f1.shape)  torch.Size([1, 4, 256, 64, 120])
        # print("f2", f2.shape) torch.Size([1, 4, 256, 128, 240])




        # calculate nerf loss
        criterion_nerf = NeRFLoss().cuda()
        if self.training:
            # nerf train & loss
            preds_rays = self.nerf_method(meta, self.model, ray_batches, device, feats2d)
            rgbs, gts, masks = [], [], []
            for ret in preds_rays:
                rgbs.append(ret['outputs_coarse']['rgb'])
                gts.append(ret['gt_rgb'])
                masks.append(ret['outputs_coarse']['mask'])
            loss_nerf = criterion_nerf(rgbs, gts, masks)


        # calculate 2D heatmap loss
        criterion = PerJointMSELoss().cuda()
        loss_2d = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        if targets_2d is not None:
            for t, w, o in zip(targets_2d, weights_2d, all_heatmaps):
                loss_2d += criterion(o, t, True, w)
            loss_2d /= len(all_heatmaps)

        loss_3d = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        if self.USE_GT:
            num_person = meta[0]['num_person']
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
            grid_centers[:, :, 0:3] = meta[0]['roots_3d'].float()
            grid_centers[:, :, 3] = -1.0
            for i in range(batch_size):
                grid_centers[i, :num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                grid_centers[i, :num_person[i], 4] = 1.0
        else:
            root_cubes, grid_centers = self.root_net(all_heatmaps, meta, self.model, feats2d)

            # calculate 3D heatmap loss
            if targets_3d is not None:
                loss_3d = criterion(root_cubes, targets_3d)
            del root_cubes

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)  # matched gt

        loss_cord = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        criterion_cord = PerJointL1Loss().cuda()
        count = 0

        for n in range(self.num_cand):
            index = (pred[:, n, 0, 3] >= 0)
            if torch.sum(index) > 0:
                single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n], self.model, feats2d)
                pred[:, n, :, 0:3] = single_pose.detach()

                # calculate 3D pose loss
                if self.training and 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
                    gt_3d = meta[0]['joints_3d'].float()
                    for i in range(batch_size):
                        if pred[i, n, 0, 3] >= 0:
                            targets = gt_3d[i:i + 1, pred[i, n, 0, 3].long()]
                            weights_3d = meta[0]['joints_3d_vis'][i:i + 1, pred[i, n, 0, 3].long(), :, 0:1].float()
                            count += 1
                            loss_cord = (loss_cord * (count - 1) +
                                         criterion_cord(single_pose[i:i + 1], targets, True, weights_3d)) / count
                del single_pose
        if self.training:
            return pred, all_heatmaps, grid_centers, loss_2d, loss_3d, loss_cord, loss_nerf
        else:
            return pred, all_heatmaps, grid_centers


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model

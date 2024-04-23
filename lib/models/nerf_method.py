import torch
import torch.nn as nn
import numpy as np

from models.render_ray import render_rays
from models.projection import Projector

from utils.cameras import unfold_camera_param_nerf

def get_rays_new(image_size, K, R, T, ret_rays_o=False):
    # calculate the camera origin
    W, H = int(image_size[0]), int(image_size[1])
    batch = K.size(0)#2
    K = K.reshape(-1, 3, 3).float()#[2,3,3]
    R = R.reshape(-1, 3, 3).float()#[2,3,3]
    T = T.reshape(-1, 3, 1).float()#[2,3,1]

    rays_o = -torch.bmm(R.transpose(2, 1), T)
    # calculate the world coordinates of pixels
    j, i = torch.meshgrid(torch.linspace(0, H-1, H),
                          torch.linspace(0, W-1, W))
    xy1 = torch.stack([i.to(K.device), j.to(K.device),
                       torch.ones_like(i).to(K.device)], dim=-1).unsqueeze(0)



    pixel_camera = torch.bmm(xy1.flatten(1, 2).repeat(batch, 1, 1),
                             torch.inverse(K).transpose(2, 1))#[10,491520,3]
    pixel_world = torch.bmm(pixel_camera-T.transpose(2, 1), R)#[10,491520,3]
    rays_d = pixel_world - rays_o.transpose(2, 1)#[10,491520,3]
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = rays_o.unsqueeze(1).repeat(1, H*W, 1, 1)
    if ret_rays_o:
        return rays_d.reshape(batch, H, W, 3), \
               rays_o.reshape(batch, H, W, 3)
    else:
        return rays_d.reshape(batch, H, W, 3)


class NerfMethodNet(nn.Module):
    def __init__(self, cfg):
        super(NerfMethodNet, self).__init__()

        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.projector = Projector(cfg)


    def collate_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)

    def forward(self, meta, nerf_model, ray_batches, device, feats2d=None):
        # meta[{}*4]  ray_bt:[{}]
        nview = len(meta)
        denorm_imgs = [metan['denorm_img'] for metan in meta]
        denorm_imgs = torch.stack(denorm_imgs, dim=1) # [bs, nview, h, w, rgb]
        batch_size = denorm_imgs.shape[0]
        cameras = [metan['camera'] for metan in meta] # [{cam0*bs}, *4]

        nerf_denorm_img = ray_batches[0]['denorm_img']
        nerf_camera = ray_batches[0]['camera']
        cam_R, cam_T, cam_K = unfold_camera_param_nerf(nerf_camera, device) # [bs, 3, 3or1]
        affine = torch.eye(3, 3, device=device)
        affine[0:2] = ray_batches[0]['transform'][0]
        affine_trans = affine.repeat(batch_size, 1, 1)
        cam_K_crop = torch.bmm(affine_trans, cam_K).view(batch_size, 3, 3)
        # get pos embed, camera ray or 2d coords
        # this can be compute only once, without iterating over views
        camera_rays_d, camera_rays_o = get_rays_new(self.img_size, cam_K_crop, cam_R, cam_T, True) # [bs, h, w, 3D]

        if feats2d is not None:
            feats2d = torch.stack(feats2d, dim=1) # [bs, nview, num_joint, heat h ,heat w]

        rgb_preds = []
        for i in range(batch_size):
            ray_batch = {
                'ray_o': camera_rays_o[i].view(-1, 3),
                'ray_d': camera_rays_d[i].view(-1, 3),
                'gt_rgb': nerf_denorm_img[i].view(-1, 3)
                }
            ret = render_rays(
                ray_batch=ray_batch,
                img=denorm_imgs[i],
                aabb=None,
                features_2D=feats2d[i] if feats2d is not None else None,
                near_far_range=[200.0, 8000.0],
                N_samples=64,
                N_rand=2048,
                nerf_mlp=nerf_model,
                img_meta=(cameras, i), # [{cam0*bs}, *4]
                projector=self.projector,
                mode='image',
                nerf_sample_view=nview,
                N_importance=0,
                is_train=True,
                render_testing=False,
                resize_transform=ray_batches[0]['transform'][0]
            )
            if ret is not None:
                rgb_preds.append(ret)

        return rgb_preds



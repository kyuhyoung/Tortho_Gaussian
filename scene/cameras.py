#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.colmap_loader import rotmat2qvec, qvec2rotmat

class SimpleCamera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image_name, uid, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(SimpleCamera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX = self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(
            0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 ):
        #qvec = rotmat2qvec(R);  qvec_inv = rotmat2qvec(R.T);  print(f'image_name : {image_name}, \nqvec : {qvec}, \nqvec_inv : {qvec_inv}, \nT : {T}');  exit() 
        #   'T' is the same as colmap output. 'qvec_inv', that is 'R.T' is the same as colmap output
        #   800779, qvec : (0.0016 -0.7074 0.7067 -0.0016), qvec_inv : (0.0016 0.7074 -0.7067 0.0016), T : (-250, 766, 1373)
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        print(f'image_name : {image_name}')
        t0 = getWorld2View2(R, T, trans, scale)
        # Since getWorld2View2 make a 4x4 matriix with the given translation T and the inverse of the given rotation R, it is the same as making a 4x4 matriix with the colmap output. 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        #   world_view_transform is the right-multiply form of 4x4 transform matirix made of colmap extrinsic params.
        #print(f't0 : \n{t0}, \nself.world_view_transform : \n{self.world_view_transform}');   exit()    #   
        self.projection_matrix = getProjectionMatrix(znear = self.znear, zfar = self.zfar, fovX = self.FoVx, fovY = self.FoVy).transpose(0, 1).cuda()
        #   projection_matrix is the right-multiply form of 4x4 matrix made of colmap focal length and image shape
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def __str__(self):
        return (f"Camera(uid={self.uid}, colmap_id={self.colmap_id}, \n"
                f"position={self.T}, \nrotation={self.R}, \n"
                f"FoVx={self.FoVx}, FoVy={self.FoVy}, image_name={self.image_name})")


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

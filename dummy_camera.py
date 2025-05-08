from utils.graphics_utils import getWorld2View2, getOrthographicProjectionMatrix
import torch
import numpy as np

class DummyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H):
        self.projection_matrix = getOrthographicProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy

class DummyPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

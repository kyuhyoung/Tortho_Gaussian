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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    #aa = bb
    # With translate (0, 0, 0) and scale 1, getWorld2View2 make a 4x4 matriix with the given translation T and the inverse of the given rotation R. 
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    #iprint(f'\tRt b4 :\n {Rt}')
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    '''
    print(f'\ttranslate : {translate}, scale : {scale}')
    print(f'\tRt after :\n {Rt}');    #exit()
    '''
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    #print(f'znear : {znear}, zfar : {zfar}, fovX : {fovX}, fovY : {fovY}');   exit()    
    # 0.01, 100, 0.6727, 0.9827   
    # fovX is 2 * atan(image_width / (2 * focal_length)). So for the case of image 800779.tif, fovX is 0.6727 = 2 * atan(11310 / (2 * 16173.8))  
    tanHalfFovY = math.tan((fovY / 2))  #   = width / (2 * focal_length), that is half width when focal length is 1
    tanHalfFovX = math.tan((fovX / 2))  #   = height / (2 * focal_length), that is half height when focal length is 1

    top = tanHalfFovY * znear   #   half height when focal length is znear   
    bottom = -top
    right = tanHalfFovX * znear #   half width when focal length is znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

######
def getOrthographicProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)    #   = half_height / focal_length
    tanHalfFovX = math.tan(fovX / 2)    #   = half_width / focal_length

    # top = tanHalfFovY * zfar / 100
    # bottom = -top
    # right = tanHalfFovX * zfar / 100
    # left = -right
    top = 6.0
    bottom = -top
    right = tanHalfFovX / tanHalfFovY * 6.0 #   If fact "tanHalfFovX / tanHalfFovY" is the same as "half_width / half_height" that is again the same as "width / height", that is the aspect ratio.  Since "fovX" and "fovY" are only used to get "right", we do not need "fovX" and "fovY".  We just the need the aspect ratio or "width" and "height" of the image we.
    left = -right
    # top = tanHalfFovY * znear
    # bottom = -top
    # right = tanHalfFovX * znear
    # left = -right
    P = torch.zeros(4, 4)

    z_sign = 1.0
    P[0, 0] = 2.0 / (right - left)
    P[0, 3] = - (right + left) / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[1, 3] = - (top + bottom) / (top - bottom)
    P[2, 2] = -2.0 / (zfar - znear)
    P[2, 3] = - (zfar + znear) / (zfar - znear)
    P[3, 3] = z_sign
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    # focal=focal*(4/3)
    return 2*math.atan(pixels/(2*focal))

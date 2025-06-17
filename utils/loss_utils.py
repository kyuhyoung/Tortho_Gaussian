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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()



def create_separable_window(window_size: int, channel: int, sigma: float = 1.5):
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    gauss  = torch.exp(-(coords**2) / (2 * sigma * sigma))
    gauss /= gauss.sum()

    v = gauss.view(1, 1, window_size, 1)   # (1,1,kH,1)
    h = gauss.view(1, 1, 1, window_size)   # (1,1,1,kW)

    # depthwise kernels
    vert  = v.repeat(channel, 1, 1, 1)     # (C,1,kH,1)
    horiz = h.repeat(channel, 1, 1, 1)     # (C,1,1,kW)
    return vert, horiz

def ssim_separable(img1, img2, window_size=11, size_average=True):
    """
    Compute SSIM between img1 and img2 using a separable Gaussian window.
    Accepts inputs of shape (C,H,W) or (N,C,H,W).
    Returns a scalar (mean SSIM) or 1D tensor of length N if size_average=False.
    """
    # if user passed (C,H,W), add batch dim
    is_3d = (img1.dim() == 3)
    if is_3d:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # now both are (N,C,H,W)
    N, C, H, W = img1.shape

    # create the two 1D kernels
    vert, horiz = create_separable_window(window_size, C)
    if img1.is_cuda:
        dev = img1.get_device()
        vert, horiz = vert.cuda(dev), horiz.cuda(dev)
    vert, horiz = vert.type_as(img1), horiz.type_as(img1)

    # compute SSIM
    ssim_map = _ssim_sep(img1, img2, vert, horiz, window_size, C)

    # average
    if size_average:
        out = ssim_map.mean()
    else:
        out = ssim_map.view(N, -1).mean(dim=1)

    # if we added a batch dim, remove it
    if is_3d:
        return out.item() if size_average else out[0]
    return out

def _ssim_sep(img1, img2, vert, horiz, window_size, channel):
    pad = window_size // 2

    # 1) local means
    mu1 = F.conv2d(img1, vert,  padding=(pad,0), groups=channel)
    mu1 = F.conv2d(mu1,   horiz, padding=(0,pad), groups=channel)
    mu2 = F.conv2d(img2, vert,  padding=(pad,0), groups=channel)
    mu2 = F.conv2d(mu2,   horiz, padding=(0,pad), groups=channel)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # 2) variances and covariance
    sigma1_sq = F.conv2d(img1*img1, vert,  padding=(pad,0), groups=channel)
    sigma1_sq = F.conv2d(sigma1_sq, horiz, padding=(0,pad), groups=channel) - mu1_sq

    sigma2_sq = F.conv2d(img2*img2, vert,  padding=(pad,0), groups=channel)
    sigma2_sq = F.conv2d(sigma2_sq, horiz, padding=(0,pad), groups=channel) - mu2_sq

    sigma12 = F.conv2d(img1*img2, vert,  padding=(pad,0), groups=channel)
    sigma12 = F.conv2d(sigma12, horiz, padding=(0,pad), groups=channel) - mu1_mu2

    # 3) SSIM map
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
               ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))

    return ssim_map



def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    #print(f'img1.shape : {img1.shape}, img2.shape : {img2.shape}, window.shape : {window.shape}, window_size : {window_size}')
    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    '''
    t0 = ssim_map.mean();   t1 = ssim_map.mean(1).mean(1).mean(1);  print(f't0 : {t0}, t1 : {t1}')
    print(f'size_average : {size_average}, ssim_map.shape : {ssim_map.shape}, ssim_map.min().item() : {ssim_map.min().item()}, ssim_map.max().item() : {ssim_map.max().item()}'); 
    '''
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_on_random_patches(
    img1: torch.Tensor,
    img2: torch.Tensor,
    patch_size: int = 64,
    n_patches: int = 16,
    window_size: int = 11,
    #patch_size: int = 6,
    #n_patches: int = 1,
    #window_size: int = 3,
    size_average: bool = False,
    separable: bool = False
) -> torch.Tensor:
    """
    img1, img2: (C, H, W) or (1, C, H, W) on CUDA
    Returns: scalar SSIM loss = 1 – mean(SSIM) if you want a loss,
             or mean SSIM directly.
    """
    # ensure 4D: (1, C, H, W)
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    _, C, H, W = img1.shape

    # choose which SSIM to call
    #print(f'separable : {separable}');  
    ssim_fn = ssim_separable if separable else ssim

    scores = []
    for iP in range(n_patches):
        top  = torch.randint(0, H - patch_size + 1, (1,), device=img1.device).item()
        left = torch.randint(0, W - patch_size + 1, (1,), device=img1.device).item()

        p1 = img1[:, :, top:top+patch_size, left:left+patch_size]
        p2 = img2[:, :, top:top+patch_size, left:left+patch_size]

        # compute SSIM on the patch
        score = ssim_fn(p1, p2,
                        window_size=window_size,
                        size_average=size_average)
        #print(f'iP : {iP}, score : {score}');  #exit()
        scores.append(score)

    # stack and average
    scores = torch.stack(scores, dim=0)  # shape (n_patches,) or (n_patches, batch)
    mean_ssim = scores.mean()
    #print(f'mean_ssim : {mean_ssim}, type(mean_ssim) : {type(mean_ssim)}');  exit()
    return mean_ssim

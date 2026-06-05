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
from collections import OrderedDict
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch

# Bounded LRU cache of decoded+resized images, kept on CPU and keyed by
# (image_path, width, height). For the multi-view-video 4D datasets there can be
# tens of thousands of images, so eager GPU upload is impossible; cameras load
# lazily on first access and the most recently used images stay resident.
_IMAGE_CACHE = OrderedDict()
_IMAGE_CACHE_CAP = 64


def _cache_get(key):
    img = _IMAGE_CACHE.get(key)
    if img is not None:
        _IMAGE_CACHE.move_to_end(key)
    return img


def _cache_put(key, tensor):
    _IMAGE_CACHE[key] = tensor
    _IMAGE_CACHE.move_to_end(key)
    while len(_IMAGE_CACHE) > _IMAGE_CACHE_CAP:
        _IMAGE_CACHE.popitem(last=False)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 timestamp: float = 0.0, frame_idx: int = 0,
                 image_path: str = None, resolution=None,
                 gt_width: int = None, gt_height: int = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # Temporal attributes (TD-FastGS 4D extension).
        # timestamp is the normalized time in [0, 1]; frame_idx is the integer frame index.
        self.timestamp = timestamp
        self.frame_idx = frame_idx

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Lazy-loading state (used when `image` is None).
        self.image_path = image_path
        self.resolution = resolution          # (W, H) target for PILtoTorch
        self._gt_alpha_mask = gt_alpha_mask    # only meaningful in the eager path

        if image is not None:
            # Eager path (3D Colmap / Blender readers): keep the original behavior.
            self._eager_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self._eager_image.shape[2]
            self.image_height = self._eager_image.shape[1]
            if gt_alpha_mask is not None:
                self._eager_image = self._eager_image * gt_alpha_mask.to(self.data_device)
            else:
                self._eager_image = self._eager_image * torch.ones(
                    (1, self.image_height, self.image_width), device=self.data_device)
        else:
            # Lazy path: dims come from the (post-resize) gt_width/gt_height.
            self._eager_image = None
            assert image_path is not None and gt_width is not None and gt_height is not None, \
                "Lazy Camera requires image_path, gt_width, gt_height"
            self.image_width = gt_width
            self.image_height = gt_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def original_image(self):
        """Return the GT image on the camera's device.

        Eager cameras hold the tensor directly. Lazy cameras decode + resize from
        disk on first access, serving repeats from a bounded CPU LRU cache so the
        full image set never has to live in memory at once.
        """
        if self._eager_image is not None:
            return self._eager_image

        key = (self.image_path, self.image_width, self.image_height)
        cpu_img = _cache_get(key)
        if cpu_img is None:
            pil = Image.open(self.image_path)
            resized = PILtoTorch(pil, self.resolution)  # (C, H, W) in [0, 1]
            rgb = resized[:3, ...].clamp(0.0, 1.0)
            if resized.shape[0] == 4:
                rgb = rgb * resized[3:4, ...]
            cpu_img = rgb.contiguous()
            _cache_put(key, cpu_img)
        return cpu_img.to(self.data_device)

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


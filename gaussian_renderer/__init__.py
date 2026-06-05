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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

def render_fastgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult, scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts}


def render_4d(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult,
              scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None):
    """TD-FastGS 4D render.

    Mandatory ordering:
      1. Spatio-temporal transform: translate centers to the current frame.
      2. Causal pruning: build the alive_mask sparse subset.
      3. Compact Box + rasterization on the alive subset only (FastGS CB runs
         inside the CUDA kernel, so subsetting the inputs is what guarantees CB
         is computed only over alive Gaussians).
      4. Back-fill per-Gaussian outputs (radii, metric counts) to full size so the
         FastGS VCD/VCP statistics keep operating on full-size tensors.

    The full-size `screenspace_points` is indexed to form the subset means2D; the
    rasterizer backward therefore scatters the screen-space gradient back into the
    full-size tensor, so add_densification_stats works exactly as in the 3D path.
    """
    t = float(viewpoint_camera.timestamp)

    # --- Step 1: spatio-temporal transform (kept in graph; w_t feeds sigma_t_raw) ---
    w_t = pc.compute_temporal_weight(t)                       # (N,)
    dt = t - pc._t_mu                                         # (N,)
    xyz_transformed = pc.get_xyz + pc._velocity * dt.unsqueeze(-1)  # (N, 3)
    opacity_eff = pc.get_opacity.squeeze(-1) * w_t           # (N,)

    N = pc.get_xyz.shape[0]

    # --- Step 2: causal pruning (boolean mask; no grad needed for the mask) ---
    with torch.no_grad():
        causal_mask = pc._t_mu <= (t + 1e-6)                 # (N,) bool
        alive_mask = causal_mask & (opacity_eff > pc.tau_alive)
        alive_idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)

    # Full-size screen-space tensor; subset rows receive grad via index backward.
    screenspace_points = torch.zeros((N, 4), dtype=pc.get_xyz.dtype,
                                     requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    radii_full = torch.zeros(N, dtype=torch.int, device="cuda")
    accum_full = torch.zeros(N, dtype=torch.int, device="cuda")

    if alive_idx.numel() == 0:
        # Nothing alive at this time: return a background image and empty stats.
        H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
        rendered_image = bg_color.view(3, 1, 1).expand(3, H, W).contiguous()
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": (radii_full > 0).nonzero(),
                "radii": radii_full,
                "accum_metric_counts": accum_full,
                "w_t": w_t,
                "alive_mask": alive_mask}

    # Rasterization configuration.
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map is None:
        metric_map = torch.zeros(int(viewpoint_camera.image_height) * int(viewpoint_camera.image_width),
                                 dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult=mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map=metric_map
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # --- Step 3: extract the alive subset of every per-Gaussian input ---
    means3D = xyz_transformed[alive_idx]
    means2D = screenspace_points[alive_idx]      # grad scatters back to full size
    opacity = opacity_eff[alive_idx].unsqueeze(-1)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[alive_idx]
    else:
        scales = pc.get_scaling[alive_idx]
        rotations = pc.get_rotation[alive_idx]

    dc = None
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)[alive_idx]
            dir_pp = (xyz_transformed[alive_idx] - viewpoint_camera.camera_center.repeat(alive_idx.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc = pc.get_features_dc[alive_idx]
            shs = pc.get_features_rest[alive_idx]
    else:
        colors_precomp = override_color[alive_idx]

    rendered_image, radii_sparse, accum_sparse = rasterizer(
        means3D=means3D,
        means2D=means2D,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # --- Step 4: back-fill per-Gaussian outputs to full size ---
    radii_full[alive_idx] = radii_sparse
    if accum_sparse is not None and accum_sparse.numel() == alive_idx.numel():
        accum_full[alive_idx] = accum_sparse.to(accum_full.dtype)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii_full > 0).nonzero(),
            "radii": radii_full,
            "accum_metric_counts": accum_full,
            "w_t": w_t,
            "alive_mask": alive_mask}
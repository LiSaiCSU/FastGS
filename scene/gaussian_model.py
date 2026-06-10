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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # ----- TD-FastGS temporal attributes -----
        # _t_mu and is_static are resident (NOT in the optimizer); _sigma_t_raw and
        # _velocity are optimized but the static rows are hard-zeroed every step.
        self.is_4d = False                 # toggled on by create_from_pcd_4d / load_ply
        self._t_mu = torch.empty(0)        # (N,)  birth time, frozen
        self._sigma_t_raw = torch.empty(0) # (N,)  life radius (log space), learnable
        self._velocity = torch.empty(0)    # (N,3) motion velocity, learnable (dynamic)
        self.is_static = torch.empty(0, dtype=torch.bool)  # (N,) identity mask
        self.tau_alive = 0.005             # causal pruning threshold on alpha'(t)
        self.n_frames = 1                  # number of temporal frames
        self._current_wt_mean = torch.empty(0)  # cached batch-mean w_t for ADC gating

        self.setup_functions()

    def capture(self, optimizer_type):
        if optimizer_type == "default":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
        else:
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _init_temporal_static(self, N):
        """Default temporal attributes for a fully-static model (3D fallback)."""
        self._t_mu = torch.zeros(N, device="cuda")
        self._sigma_t_raw = nn.Parameter(
            torch.full((N,), math.log(1000.0), device="cuda").requires_grad_(True))
        self._velocity = nn.Parameter(
            torch.zeros((N, 3), device="cuda").requires_grad_(True))
        self.is_static = torch.ones(N, dtype=torch.bool, device="cuda")

    def compute_temporal_weight(self, t):
        """Per-Gaussian temporal activity weight w_t^(i)(t), shape (N,).

        w_t = exp(-(t - t_mu)^2 / (2 sigma_t^2 + eps)); static points are pinned
        to 1.0. Must NOT be wrapped in torch.no_grad() when its gradient w.r.t.
        sigma_t_raw is needed (rendering / VCD score)."""
        sigma_t = torch.exp(self._sigma_t_raw)          # (N,)
        dt = t - self._t_mu                             # (N,)
        w_t = torch.exp(-dt ** 2 / (2 * sigma_t ** 2 + 1e-8))
        # Pin static points to 1.0 without breaking the graph for dynamic points.
        w_t = torch.where(self.is_static, torch.ones_like(w_t), w_t)
        return w_t

    def create_from_pcd_4d(self, tpcd, spatial_lr_scale, n_frames):
        """Initialize Gaussians from a TemporalPointCloud (static-first ordering).

        Static points: t_mu=0, sigma_t=1000 (full-time visible), v=0, frozen.
        Dynamic points: t_mu=birth timestamp, sigma_t covering ~2.5 frames, v=0.
        """
        self.is_4d = True
        self.n_frames = max(int(n_frames), 1)
        self.spatial_lr_scale = spatial_lr_scale

        points = np.asarray(tpcd.points)
        colors = np.asarray(tpcd.colors)
        timestamps = np.asarray(tpcd.timestamps)
        is_static_np = np.asarray(tpcd.is_static)

        fused_point_cloud = torch.tensor(points).float().cuda()
        fused_color = RGB2SH(torch.tensor(colors).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        N = fused_point_cloud.shape[0]
        print(f"[TD-FastGS] points at init: {N} "
              f"(static={int(is_static_np.sum())}, dynamic={int((~is_static_np).sum())})")

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((N), device="cuda")

        # Temporal attributes.
        is_static = torch.tensor(is_static_np, dtype=torch.bool, device="cuda")
        t_mu = torch.tensor(timestamps, dtype=torch.float, device="cuda")
        # Static points: t_mu pinned to 0.
        t_mu = torch.where(is_static, torch.zeros_like(t_mu), t_mu)

        sigma_t_raw = torch.empty(N, device="cuda")
        sigma_t_raw[is_static] = math.log(1000.0)
        sigma_t_raw[~is_static] = math.log(2.5 / max(self.n_frames, 1))

        # Use flow-estimated velocities if available, otherwise zero-init.
        vel_np = getattr(tpcd, "velocities", None)
        if vel_np is not None and np.asarray(vel_np).shape == (N, 3):
            velocity = torch.tensor(np.asarray(vel_np), dtype=torch.float, device="cuda")
            print(f"[TD-FastGS] initialising velocity from optical flow "
                  f"(nonzero: {int((velocity.norm(dim=1) > 1e-6).sum())}/{N})")
        else:
            velocity = torch.zeros((N, 3), device="cuda")

        self.is_static = is_static
        self._t_mu = t_mu
        self._sigma_t_raw = nn.Parameter(sigma_t_raw.requires_grad_(True))
        self._velocity = nn.Parameter(velocity.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        # Temporal parameters share the main optimizer so prune/clone/cat keep them
        # in sync with the geometry tensors (writeback is keyed on group["name"]).
        if self.is_4d:
            if not isinstance(self._sigma_t_raw, nn.Parameter):
                self._init_temporal_static(self.get_xyz.shape[0])
            l.append({'params': [self._velocity],
                      'lr': getattr(training_args, "velocity_lr", 0.0016), "name": "velocity"})
            l.append({'params': [self._sigma_t_raw],
                      'lr': getattr(training_args, "sigma_t_lr", 0.002), "name": "sigma_t_raw"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.is_4d:
            l += ['t_mu', 'sigma_t_raw', 'vel_x', 'vel_y', 'vel_z', 'is_static']
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.is_4d:
            t_mu = self._t_mu.detach().cpu().numpy()[:, None]
            sigma_t_raw = self._sigma_t_raw.detach().cpu().numpy()[:, None]
            velocity = self._velocity.detach().cpu().numpy()
            is_static = self.is_static.detach().cpu().numpy().astype(np.float32)[:, None]
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation,
                                         t_mu, sigma_t_raw, velocity, is_static), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_decoupled(self, reset_value=0.01):
        """Decoupled opacity reset (TD-FastGS): only static points are reset; dynamic
        points keep their opacity to protect foreground temporal state. Adam state is
        synced via replace_tensor_to_optimizer."""
        if not self.is_4d:
            return self.reset_opacity()
        with torch.no_grad():
            static_mask = self.is_static
            target = self.inverse_opacity_activation(
                torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * reset_value))
            opacities_new = self._opacity.clone()
            opacities_new[static_mask] = target[static_mask]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def enforce_static_constraints(self):
        """Hard pull-back for static points to counter Adam momentum residue.
        Call immediately after optimizer.step(). Zeros velocity, pins sigma_t_raw to
        log(1000) and t_mu to 0, and clears the matching Adam moments."""
        if not self.is_4d:
            return
        with torch.no_grad():
            static_mask = self.is_static
            if static_mask.sum() == 0:
                return
            self._velocity.data[static_mask] = 0.0
            self._sigma_t_raw.data[static_mask] = math.log(1000.0)
            self._t_mu[static_mask] = 0.0
            for group in self.optimizer.param_groups:
                if group["name"] in ("velocity", "sigma_t_raw"):
                    p = group["params"][0]
                    state = self.optimizer.state.get(p, None)
                    if state is not None and "exp_avg" in state:
                        state["exp_avg"][static_mask] = 0.0
                        state["exp_avg_sq"][static_mask] = 0.0

    def apply_gradient_gating(self, t_current, wt_current_thresh=0.5):
        """Three-level gradient gate (call after backward(), before step()):
          - static points: velocity & sigma_t_raw grads zeroed;
          - dynamic & current (w_t > thresh): all grads pass;
          - dynamic & other frame: geometry grads zeroed, opacity/velocity/sigma_t pass."""
        if not self.is_4d:
            return
        with torch.no_grad():
            w_t = self.compute_temporal_weight(t_current).detach()
            is_static = self.is_static
            is_dynamic_other = (~is_static) & (w_t <= wt_current_thresh)

            for name in ("_velocity", "_sigma_t_raw"):
                p = getattr(self, name)
                if p.grad is not None:
                    p.grad[is_static] = 0.0

            for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation"):
                p = getattr(self, name)
                if p.grad is not None:
                    p.grad[is_dynamic_other] = 0.0

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Restore temporal attributes if present; otherwise degrade to static mode.
        prop_names = [p.name for p in plydata.elements[0].properties]
        N = xyz.shape[0]
        if "t_mu" in prop_names and "sigma_t_raw" in prop_names and "vel_x" in prop_names:
            self.is_4d = True
            t_mu = np.asarray(plydata.elements[0]["t_mu"])
            sigma_t_raw = np.asarray(plydata.elements[0]["sigma_t_raw"])
            vel = np.stack((np.asarray(plydata.elements[0]["vel_x"]),
                            np.asarray(plydata.elements[0]["vel_y"]),
                            np.asarray(plydata.elements[0]["vel_z"])), axis=1)
            if "is_static" in prop_names:
                is_static = np.asarray(plydata.elements[0]["is_static"]) > 0.5
            else:
                is_static = np.zeros(N, dtype=bool)
            self._t_mu = torch.tensor(t_mu, dtype=torch.float, device="cuda")
            self._sigma_t_raw = nn.Parameter(torch.tensor(sigma_t_raw, dtype=torch.float, device="cuda").requires_grad_(True))
            self._velocity = nn.Parameter(torch.tensor(vel, dtype=torch.float, device="cuda").requires_grad_(True))
            self.is_static = torch.tensor(is_static, dtype=torch.bool, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.is_4d:
            self._velocity = optimizable_tensors["velocity"]
            self._sigma_t_raw = optimizable_tensors["sigma_t_raw"]
            # Resident (non-optimizer) temporal tensors.
            self._t_mu = self._t_mu[valid_points_mask]
            self.is_static = self.is_static[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii,
                              new_velocity=None, new_sigma_t_raw=None, new_t_mu=None, new_is_static=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        if self.is_4d:
            d["velocity"] = new_velocity
            d["sigma_t_raw"] = new_sigma_t_raw

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.is_4d:
            self._velocity = optimizable_tensors["velocity"]
            self._sigma_t_raw = optimizable_tensors["sigma_t_raw"]
            # Resident temporal tensors grow by concatenation.
            self._t_mu = torch.cat((self._t_mu, new_t_mu), dim=0)
            self.is_static = torch.cat((self.is_static, new_is_static), dim=0)

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        if self.is_4d:
            # Children inherit temporal attributes verbatim (position is perturbed,
            # temporal parameters are copied). Static children stay static.
            new_velocity = self._velocity[selected_pts_mask].repeat(N, 1)
            new_sigma_t_raw = self._sigma_t_raw[selected_pts_mask].repeat(N)
            new_t_mu = self._t_mu[selected_pts_mask].repeat(N)
            new_is_static = self.is_static[selected_pts_mask].repeat(N)
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii,
                                       new_velocity=new_velocity, new_sigma_t_raw=new_sigma_t_raw,
                                       new_t_mu=new_t_mu, new_is_static=new_is_static)
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        if self.is_4d:
            new_velocity = self._velocity[selected_pts_mask]
            new_sigma_t_raw = self._sigma_t_raw[selected_pts_mask]
            new_t_mu = self._t_mu[selected_pts_mask]
            new_is_static = self.is_static[selected_pts_mask]
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii,
                                       new_velocity=new_velocity, new_sigma_t_raw=new_sigma_t_raw,
                                       new_t_mu=new_t_mu, new_is_static=new_is_static)
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score = None, pruning_score = None):
        
        ''' 
            Densification and Pruning based on FastGS criteria:
            1.  The gaussians candidate for densification are selected based on the gradient of their position first.
            2.  Then, based on their average metric score (computed over multiple sampled views), they are either densified (cloned) or split.
                This is our main contribution compared to the vanilla 3DGS.
            3.  Finally, gaussians with low opacity or very large size are pruned.
        '''
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # This is our multi-view consisent metric for densification
        # We use this metric to further filter the candidates for densification, which is similar to taming 3dgs.
        metric_mask = importance_score > 5

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        scores = 1 - pruning_score 
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)

        # The budget is not necessary for our method.
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            self.prune_points(final_prune)
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def final_prune_fastgs(self, min_opacity, pruning_score = None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune)

    # ===================== TD-FastGS temporal ADC =====================

    def set_current_wt_mean(self, timestamps):
        """Cache the per-Gaussian mean temporal weight over a set of view timestamps.
        Used by the densify/prune gating to decide which dynamic points are 'active'
        in the current densification batch. Computed without grad."""
        with torch.no_grad():
            if len(timestamps) == 0:
                self._current_wt_mean = torch.ones(self.get_xyz.shape[0], device="cuda")
                return
            acc = torch.zeros(self.get_xyz.shape[0], device="cuda")
            for t in timestamps:
                acc += self.compute_temporal_weight(float(t))
            self._current_wt_mean = acc / len(timestamps)

    def densify_and_prune_4d(self, max_screen_size, min_opacity, extent, radii, args,
                             importance_score=None, pruning_score=None):
        """Temporal-aware ADC. Mirrors densify_and_prune_fastgs but:
          - uses per-point densification thresholds (static=tau_d_static,
            dynamic=tau_d_dynamic) and gates dynamic densify on the active window;
          - prunes static points by VCP and dynamic points by credit-assigned VCP
            restricted to their active window (w_t > wt_densify_thresh)."""
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.norm(grad_vars, dim=-1) >= args.grad_thresh
        grad_qualifiers_abs = torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense * extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense * extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # Per-point densification threshold (dynamic points use a lower tau_d).
        tau_d = torch.where(self.is_static,
                            torch.full_like(self._current_wt_mean, args.tau_d_static),
                            torch.full_like(self._current_wt_mean, args.tau_d_dynamic))
        metric_mask = importance_score.squeeze() > tau_d

        # Dynamic points may only densify inside their active window.
        dynamic_active = (~self.is_static) & (self._current_wt_mean > args.wt_densify_thresh)
        densify_allowed = self.is_static | dynamic_active
        metric_mask = metric_mask & densify_allowed

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        # ---- pruning ----
        # Clone/split appended new points at the end, so vcp / wt_mean (computed at
        # the pre-densification size) must be padded to the current size. New points
        # get score 0 (eligible for opacity pruning only, never VCP pruning).
        N_now = self.get_xyz.shape[0]
        vcp = pruning_score.squeeze()
        if vcp.shape[0] < N_now:
            pad = torch.zeros(N_now - vcp.shape[0], device=vcp.device, dtype=vcp.dtype)
            vcp = torch.cat((vcp, pad), dim=0)
        wt_mean = self._current_wt_mean
        if wt_mean.shape[0] < N_now:
            pad = torch.zeros(N_now - wt_mean.shape[0], device=wt_mean.device, dtype=wt_mean.dtype)
            wt_mean = torch.cat((wt_mean, pad), dim=0)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # VCP score prune: static always eligible; dynamic only inside active window.
        static_prune = self.is_static & (vcp > args.tau_p)
        dyn_active_now = (~self.is_static) & (wt_mean > args.wt_densify_thresh)
        dynamic_prune = dyn_active_now & (vcp > args.tau_p)
        prune_mask = prune_mask | static_prune | dynamic_prune

        self.prune_points(prune_mask)

        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def final_prune_4d(self, min_opacity, pruning_score=None, args=None):
        """Final-stage temporal pruning. Like final_prune_fastgs, but dynamic points
        are protected outside their active window (credit assignment)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        vcp = pruning_score.squeeze()
        wt_thresh = args.wt_densify_thresh if args is not None else 0.2
        static_prune = self.is_static & (vcp > 0.9)
        dyn_active = (~self.is_static) & (self._current_wt_mean > wt_thresh)
        dynamic_prune = dyn_active & (vcp > 0.9)
        final_prune = prune_mask | static_prune | dynamic_prune
        self.prune_points(final_prune)
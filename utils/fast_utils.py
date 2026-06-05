import torch
from PIL import ImageFilter
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random


def sampling_cameras(my_viewpoint_stack):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    num_cams = 10
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))

    return camlist

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY = False):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)
        
        metric_map = (l1_loss_norm > args.loss_thresh).int()

        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score


# ============================================================================
#  TD-FastGS 4D (temporal) helpers
# ============================================================================

def sample_camera_4d(train_cameras, iteration, n_frames, opt):
    """Three-stage temporal camera sampling strategy.

    Stage 1 (iter <= static_only_until): sample only frame-0 views to first
        converge the static background base.
    Stage 2 (<= temporal_window_until): pick a random start frame and sample
        within a contiguous window of `temporal_window_size` frames, giving the
        velocity term adjacent-frame supervision.
    Stage 3 (otherwise): uniform random over all training cameras.
    """
    if iteration <= opt.static_only_until:
        pool = [c for c in train_cameras if c.frame_idx == 0]
        if not pool:
            pool = train_cameras
        return random.choice(pool)
    elif iteration <= opt.temporal_window_until:
        w = max(1, opt.temporal_window_size)
        start = random.randint(0, max(0, n_frames - w))
        pool = [c for c in train_cameras if start <= c.frame_idx < start + w]
        if not pool:
            pool = train_cameras
        return random.choice(pool)
    else:
        return random.choice(train_cameras)


def sample_views_for_vcd_vcp(train_cameras, K, iteration, opt):
    """Sample K views for the temporal VCD/VCP score computation.

    Aligned with the camera-sampling strategy: stage 1 draws only frame-0 views
    (static-scene consistency); afterwards it samples globally and the temporal
    weight in the score automatically discounts inactive views. Returns a list of
    Camera objects (each carrying a `timestamp`)."""
    if iteration <= opt.static_only_until:
        pool = [c for c in train_cameras if c.frame_idx == 0]
        if not pool:
            pool = train_cameras
    else:
        pool = train_cameras
    return random.sample(pool, min(K, len(pool)))


def compute_velocity_smoothness_loss(gaussians, K_pairs=4096):
    """Spatially-weighted velocity consistency regularizer over dynamic points.

        L_smooth = (1/K) * sum_k w_k * ||v_{a_k} - v_{b_k}||^2
        w_k      = exp(-||x_{a_k} - x_{b_k}||^2 / (2 * s_bar^2))

    s_bar^2 is a global spatial scale estimated from the sampled pair distances.
    Returns a zero scalar if there are fewer than two dynamic points.
    """
    dynamic_idx = (~gaussians.is_static).nonzero(as_tuple=False).squeeze(-1)
    if dynamic_idx.shape[0] < 2:
        return torch.zeros((), device="cuda")

    n = dynamic_idx.shape[0]
    idx_a = dynamic_idx[torch.randint(0, n, (K_pairs,), device=dynamic_idx.device)]
    idx_b = dynamic_idx[torch.randint(0, n, (K_pairs,), device=dynamic_idx.device)]

    pos_a = gaussians.get_xyz[idx_a]
    pos_b = gaussians.get_xyz[idx_b]
    vel_a = gaussians._velocity[idx_a]
    vel_b = gaussians._velocity[idx_b]

    dist_sq = ((pos_a - pos_b) ** 2).sum(-1)
    s_bar_sq = dist_sq.mean().detach() + 1e-8
    w = torch.exp(-dist_sq / (2 * s_bar_sq))
    loss = (w * ((vel_a - vel_b) ** 2).sum(-1)).mean()
    return loss


def compute_gaussian_score_fastgs_4d(camlist, gaussians, pipe, bg, args, render_4d, DENSIFY=False):
    """Temporal-aware multi-view VCD/VCP score.

    Identical in structure to compute_gaussian_score_fastgs, but (1) renders each
    view through the 4D renderer at that view's timestamp, and (2) weights each
    view's per-Gaussian high-error counts by the temporal weight w_t(t_view) so
    that Gaussians inactive at a view's time contribute ~0 to that view's score.

    IMPORTANT: compute_temporal_weight must be called OUTSIDE torch.no_grad() so
    that gradients can flow to sigma_t_raw. The caller is responsible for the
    grad context; here we keep w_t in the graph and only detach where counts are
    combined into the (non-differentiable) score statistics.
    """
    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        cam = camlist[view]
        t_j = cam.timestamp

        render_pkg0 = render_4d(cam, gaussians, pipe, bg, args.mult)
        render_image = render_pkg0["render"]
        photometric_loss = compute_photometric_loss(cam, render_image)

        gt_image = cam.original_image.cuda()
        l1_loss_norm = get_loss(render_image, gt_image)
        metric_map = (l1_loss_norm > args.loss_thresh).int()

        render_pkg = render_4d(cam, gaussians, pipe, bg, args.mult,
                               get_flag=True, metric_map=metric_map)
        accum_loss_counts = render_pkg["accum_metric_counts"]  # (N_full,)

        # Temporal weight at this view's time, broadcast over the full set.
        w_t = gaussians.compute_temporal_weight(t_j).detach()  # (N,)
        weighted_counts = accum_loss_counts.float() * w_t

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = weighted_counts.clone()
            else:
                full_metric_counts += weighted_counts

        contrib = photometric_loss * weighted_counts
        if full_metric_score is None:
            full_metric_score = contrib.clone()
        else:
            full_metric_score += contrib

    denom = (torch.max(full_metric_score) - torch.min(full_metric_score))
    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (denom + 1e-8)

    if DENSIFY:
        importance_score = full_metric_counts / len(camlist)
    else:
        importance_score = None
    return importance_score, pruning_score

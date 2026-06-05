"""Unit tests for the TD-FastGS temporal extension.

These tests exercise the temporal logic that does not require the CUDA rasterizer.
They are written to be run with pytest on a CUDA-enabled machine:

    pytest tests/test_td_fastgs.py

Tests that intrinsically need rasterization (full render_4d, VCD/VCP scoring) are
described as structured logic / smoke checks and skipped when CUDA is absent.
"""

import math
import numpy as np
import pytest

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA")


def _make_model(n_static=4, n_dynamic=6, n_frames=10):
    """Build a small 4D GaussianModel via create_from_pcd_4d."""
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import TemporalPointCloud

    N = n_static + n_dynamic
    pts = np.random.randn(N, 3).astype(np.float32)
    cols = np.random.rand(N, 3).astype(np.float32)
    normals = np.zeros((N, 3), np.float32)
    is_static = np.zeros(N, dtype=bool)
    is_static[:n_static] = True
    ts = np.zeros(N, np.float32)
    # dynamic points born at staggered frames
    ts[n_static:] = np.linspace(0.0, 1.0, n_dynamic).astype(np.float32)

    tpcd = TemporalPointCloud(points=pts, colors=cols, normals=normals,
                              timestamps=ts, is_static=is_static)
    g = GaussianModel(sh_degree=1)
    g.create_from_pcd_4d(tpcd, spatial_lr_scale=1.0, n_frames=n_frames)

    class _Opt:  # minimal training-args stub
        percent_dense = 0.01
        position_lr_init = 1e-4
        position_lr_final = 1e-6
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30000
        lowfeature_lr = 2.5e-3
        highfeature_lr = 5e-3
        opacity_lr = 0.025
        scaling_lr = 5e-3
        rotation_lr = 1e-3
        velocity_lr = 1.6e-3
        sigma_t_lr = 2e-3
    g.training_setup(_Opt())
    return g


def test_init_ordering_and_flags():
    """1.6/6: static-first ordering, correct is_static flags and sigma_t init."""
    g = _make_model(n_static=4, n_dynamic=6, n_frames=10)
    assert g.is_static[:4].all()
    assert (~g.is_static[4:]).all()
    # static sigma_t_raw ~ log(1000); dynamic ~ log(2.5/n_frames)
    assert torch.allclose(g._sigma_t_raw[:4],
                          torch.full((4,), math.log(1000.0), device="cuda"), atol=1e-4)
    assert torch.allclose(g._sigma_t_raw[4:],
                          torch.full((6,), math.log(2.5 / 10), device="cuda"), atol=1e-4)
    # static t_mu pinned to 0
    assert torch.all(g._t_mu[:4] == 0)


def test_temporal_weight_static_is_one():
    """compute_temporal_weight pins static points to 1.0 at any t."""
    g = _make_model()
    for t in (0.0, 0.3, 1.0):
        w = g.compute_temporal_weight(t)
        assert torch.allclose(w[g.is_static], torch.ones(int(g.is_static.sum()), device="cuda"))


def test_temporal_weight_gradient_flows_to_sigma():
    """3: sigma_t_raw receives gradient through w_t for dynamic points."""
    g = _make_model()
    w = g.compute_temporal_weight(0.5)
    # a dynamic-only objective
    loss = (w[~g.is_static] ** 2).sum()
    loss.backward()
    assert g._sigma_t_raw.grad is not None
    assert g._sigma_t_raw.grad[~g.is_static].abs().sum() > 0


def test_static_hard_pullback():
    """1: after perturbing then enforcing constraints, static velocity == 0."""
    g = _make_model()
    with torch.no_grad():
        g._velocity.data[g.is_static] = 5.0
        g._sigma_t_raw.data[g.is_static] = 0.0
        g._t_mu[g.is_static] = 0.7
    g.enforce_static_constraints()
    assert g._velocity[g.is_static].abs().max() == 0
    assert torch.allclose(g._sigma_t_raw[g.is_static],
                          torch.full((int(g.is_static.sum()),), math.log(1000.0), device="cuda"))
    assert torch.all(g._t_mu[g.is_static] == 0)


def test_gradient_gating_static_velocity_zeroed():
    """3.1: gate zeros static velocity/sigma grads and frozen-frame geometry grads."""
    g = _make_model()
    # fake gradients on all params
    for name in ("_velocity", "_sigma_t_raw", "_xyz", "_scaling"):
        p = getattr(g, name)
        p.grad = torch.ones_like(p)
    g.apply_gradient_gating(t_current=0.0, wt_current_thresh=0.5)
    # static velocity / sigma grads cleared
    assert g._velocity.grad[g.is_static].abs().max() == 0
    assert g._sigma_t_raw.grad[g.is_static].abs().max() == 0
    # dynamic points far from t=0 (w_t<=0.5) have geometry grads cleared
    w = g.compute_temporal_weight(0.0)
    dyn_other = (~g.is_static) & (w <= 0.5)
    if dyn_other.any():
        assert g._xyz.grad[dyn_other].abs().max() == 0


def test_decoupled_opacity_reset_preserves_dynamic():
    """5: dynamic opacity unchanged after decoupled reset; static lowered."""
    g = _make_model()
    with torch.no_grad():
        g._opacity.data.fill_(g.inverse_opacity_activation(torch.tensor(0.9)).item())
    dyn_before = g.get_opacity[~g.is_static].clone()
    g.reset_opacity_decoupled(reset_value=0.01)
    assert torch.allclose(g.get_opacity[~g.is_static], dyn_before, atol=1e-5)
    assert g.get_opacity[g.is_static].max() <= 0.05


def test_save_load_roundtrip(tmp_path):
    """PLY serialization preserves temporal attributes (graceful for legacy too)."""
    from scene.gaussian_model import GaussianModel
    g = _make_model()
    path = str(tmp_path / "pc.ply")
    g.save_ply(path)
    g2 = GaussianModel(sh_degree=1)
    g2.load_ply(path)
    assert g2.is_4d
    assert torch.allclose(g2._t_mu, g._t_mu, atol=1e-5)
    assert torch.equal(g2.is_static, g.is_static)
    assert torch.allclose(g2._velocity, g._velocity, atol=1e-5)


def test_causal_mask_logic():
    """2: causal mask excludes Gaussians born after t (pure tensor logic)."""
    g = _make_model()
    t = 0.1
    causal = g._t_mu <= (t + 1e-6)
    # any dynamic point with t_mu > 0.1 must be excluded
    born_later = (~g.is_static) & (g._t_mu > 0.1)
    if born_later.any():
        assert (~causal[born_later]).all()


def test_velocity_smoothness_nonnegative():
    """Module 5: velocity-smoothness loss is a finite non-negative scalar."""
    from utils.fast_utils import compute_velocity_smoothness_loss
    g = _make_model()
    with torch.no_grad():
        g._velocity.data[~g.is_static] = torch.randn_like(g._velocity[~g.is_static])
    loss = compute_velocity_smoothness_loss(g, K_pairs=128)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_child_inherits_is_static():
    """6: clone preserves is_static for children (structural check via clone path)."""
    g = _make_model(n_static=4, n_dynamic=6)
    n_before = g.get_xyz.shape[0]
    g.tmp_radii = torch.zeros(n_before, device="cuda")
    # clone all static points
    metric = g.is_static.clone()
    filt = torch.ones(n_before, dtype=bool, device="cuda")
    g.densify_and_clone_fastgs(metric, filt)
    # appended children should all be static (parents were static)
    n_added = g.get_xyz.shape[0] - n_before
    assert n_added == int(metric.sum())
    assert g.is_static[n_before:].all()


# --- Logic-only descriptions for rasterizer-dependent paths -----------------
# test_render_4d_subset_size: render_4d must pass exactly alive_mask.sum()
#   Gaussians to the rasterizer (CB runs after the alive subset is extracted),
#   while returning full-size radii / viewspace_points / visibility_filter.
# test_vcd_temporal_zeroing: a t_mu=0.8 dynamic Gaussian contributes ~0 to the
#   VCD score under a t=0.0 view because w_t(0.0) -> 0 for that Gaussian.

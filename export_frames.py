"""
从训练好的 4DGS point_cloud.ply 中，按帧导出每一帧的高斯点云。

兼容两种属性命名：
  TD-FastGS (本项目): sigma_t_raw, vel_x/vel_y/vel_z, is_static
  DT-4DGS (原始):     t_sigma,     velocity_0/1/2

对于每个帧时间 t，执行：
  1. 位置偏移: xyz_t = xyz + velocity * (t - t_mu)
  2. 时域高斯权重: w = exp(- (t - t_mu)^2 / (2 * sigma^2 + 1e-5))
  3. 不透明度调制: opacity_t = opacity * w
  4. 因果律: 仅保留 t_mu <= t 且 opacity_t > threshold 的点

用法:
python export_frames.py --ply <path_to_point_cloud.ply> --out <output_dir> --num_frames 80
python export_frames.py --ply output/flower_1/point_cloud/iteration_30000/point_cloud.ply --out output/flower_1/frames --num_frames 80
"""

import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement


def load_4dgs_ply(path):
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    all_props = [p.name for p in v.properties]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)  # (N, 3)

    # opacity (raw, pre-sigmoid)
    opacity_raw = np.asarray(v["opacity"])  # (N,)

    # t_mu always present
    t_mu = np.asarray(v["t_mu"])            # (N,)

    # Temporal-width field: TD-FastGS saves "sigma_t_raw"; DT-4DGS saves "t_sigma".
    if "sigma_t_raw" in all_props:
        t_sigma_raw = np.asarray(v["sigma_t_raw"])
    elif "t_sigma" in all_props:
        t_sigma_raw = np.asarray(v["t_sigma"])
    else:
        raise ValueError("PLY has neither 'sigma_t_raw' nor 't_sigma' — not a 4DGS file?")

    # Velocity: TD-FastGS uses vel_x/y/z; DT-4DGS uses velocity_0/1/2.
    if "vel_x" in all_props:
        velocity = np.stack([v["vel_x"], v["vel_y"], v["vel_z"]], axis=1)
    elif "velocity_0" in all_props:
        velocity = np.stack([v["velocity_0"], v["velocity_1"], v["velocity_2"]], axis=1)
    else:
        raise ValueError("PLY has no velocity attributes — not a 4DGS file?")

    return plydata, xyz, opacity_raw, t_mu, t_sigma_raw, velocity, all_props


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def export_frame(plydata, xyz, opacity_raw, t_mu, t_sigma_raw, velocity, frame_t, threshold=0.005):
    """返回该帧存活点的索引、偏移后的 xyz、调制后的 opacity_raw"""
    dt = frame_t - t_mu                                            # (N,)
    xyz_t = xyz + velocity * dt[:, None]                           # (N, 3)

    sigma = np.exp(t_sigma_raw)                                    # (N,)
    temporal_weight = np.exp(-(dt ** 2) / (2.0 * sigma ** 2 + 1e-5))  # (N,)

    opacity_activated = sigmoid(opacity_raw)                       # (N,)
    opacity_t = opacity_activated * temporal_weight                # (N,)

    # 因果律剪枝
    alive = (t_mu <= frame_t) & (opacity_t > threshold)

    # 调制后的 opacity 转回 raw (inverse sigmoid)
    opacity_t_clamped = np.clip(opacity_t, 1e-7, 1.0 - 1e-7)
    opacity_t_raw = np.log(opacity_t_clamped / (1.0 - opacity_t_clamped))

    return alive, xyz_t, opacity_t_raw


def save_frame_ply(plydata, alive_mask, xyz_t, opacity_t_raw, output_path):
    """将存活点写成标准 3DGS PLY（去掉 4DGS 专属字段）"""
    src = plydata["vertex"]

    # Skip all 4DGS-specific fields regardless of naming convention.
    skip = {
        "t_mu",
        "sigma_t_raw", "t_sigma",               # TD-FastGS / DT-4DGS temporal width
        "vel_x", "vel_y", "vel_z",              # TD-FastGS velocity
        "velocity_0", "velocity_1", "velocity_2",  # DT-4DGS velocity
        "is_static",                            # TD-FastGS static flag
    }
    keep_names = [p.name for p in src.properties if p.name not in skip]
    src_dtype = src.data.dtype
    dtype_out = [(name, src_dtype[name].str) for name in keep_names]

    n_alive = int(alive_mask.sum())
    elements = np.empty(n_alive, dtype=dtype_out)

    for name in keep_names:
        col = np.asarray(src[name])[alive_mask]
        if name == "x":
            col = xyz_t[alive_mask, 0]
        elif name == "y":
            col = xyz_t[alive_mask, 1]
        elif name == "z":
            col = xyz_t[alive_mask, 2]
        elif name == "opacity":
            col = opacity_t_raw[alive_mask]
        elements[name] = col

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)


def main():
    parser = argparse.ArgumentParser(description="Export per-frame PLY from a 4DGS checkpoint")
    parser.add_argument("--ply", type=str, required=True, help="Path to the trained point_cloud.ply")
    parser.add_argument("--out", type=str, required=True, help="Output directory for per-frame PLY files")
    parser.add_argument("--num_frames", type=int, default=80, help="Number of frames to export")
    parser.add_argument("--threshold", type=float, default=0.005, help="Opacity threshold for pruning")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading PLY: {args.ply}")
    plydata, xyz, opacity_raw, t_mu, t_sigma_raw, velocity, all_props = load_4dgs_ply(args.ply)
    print(f"Total gaussians: {xyz.shape[0]}")

    for i in range(args.num_frames):
        frame_t = i / max(args.num_frames - 1, 1)
        alive, xyz_t, opacity_t_raw = export_frame(
            plydata, xyz, opacity_raw, t_mu, t_sigma_raw, velocity, frame_t, args.threshold
        )
        n_alive = int(alive.sum())

        out_path = os.path.join(args.out, f"{i + 1}.ply")
        save_frame_ply(plydata, alive, xyz_t, opacity_t_raw, out_path)
        print(f"Frame {i + 1}/{args.num_frames}  t={frame_t:.4f}  alive={n_alive}  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

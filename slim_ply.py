"""
对 4DGS point_cloud.ply 进行瘦身，仅保留渲染器所需属性。

兼容两种 4DGS 属性命名（输入自动检测，输出统一为 DT-4DGS 命名）：
  TD-FastGS (本项目): sigma_t_raw, vel_x/vel_y/vel_z, is_static
  DT-4DGS (原始):     t_sigma,     velocity_0/1/2

输出始终使用 DT-4DGS 字段名，与播放器兼容：
  x, y, z
  f_dc_0, f_dc_1, f_dc_2
  f_rest_0..2, f_rest_15..17, f_rest_30..32
  opacity
  scale_0, scale_1, scale_2
  rot_0, rot_1, rot_2, rot_3
  t_mu, t_sigma        ← 时域中心与宽度（标准名）
  velocity_0/1/2       ← 速度（标准名）

用法：
  python slim_ply.py                          # 默认处理 point_cloud.ply → point_cloud_slim.ply
  python slim_ply.py -i input.ply -o out.ply  # 指定输入/输出路径
"""

import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement


# Output attribute list (DT-4DGS naming — what the player expects).
_OUT_PROPS = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "f_rest_0", "f_rest_1", "f_rest_2",
    "f_rest_15", "f_rest_16", "f_rest_17",
    "f_rest_30", "f_rest_31", "f_rest_32",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "t_mu", "t_sigma",
    "velocity_0", "velocity_1", "velocity_2",
]

# Mapping: input field name → canonical output field name.
# Fields absent from this map are copied under the same name.
_RENAME = {
    "sigma_t_raw": "t_sigma",
    "vel_x":       "velocity_0",
    "vel_y":       "velocity_1",
    "vel_z":       "velocity_2",
}

# Fields that exist only in TD-FastGS and have no output equivalent → drop.
_DROP = {"is_static"}


def slim_ply(input_path: str, output_path: str, num_frames: int, vel_threshold: float = 1e-3) -> None:
    print(f"读取：{input_path}")
    plydata = PlyData.read(input_path)
    src = plydata["vertex"]
    src_dtype = src.data.dtype

    all_props = [p.name for p in src.properties]
    print(f"原始属性数：{len(all_props)}")

    # Build the output column map: output_name → source data array.
    col_data = {}   # output_name → np.ndarray
    col_dtype = {}  # output_name → dtype str
    for name in all_props:
        out_name = _RENAME.get(name, name)
        if out_name in _DROP or out_name not in _OUT_PROPS:
            continue
        col_data[out_name] = np.asarray(src.data[name])
        col_dtype[out_name] = src_dtype[name].str

    # Determine which output columns we actually have.
    out_names = [n for n in _OUT_PROPS if n in col_data]
    removed = [n for n in all_props if (_RENAME.get(n, n) not in col_data)]
    print(f"输出属性数：{len(out_names)}")
    print(f"删除/映射后丢弃属性数：{len(all_props) - len(out_names)}")
    if removed:
        print(f"删除属性：{removed}")

    # Detect which sigma/velocity source was present for informational output.
    sigma_src = "sigma_t_raw" if "sigma_t_raw" in all_props else ("t_sigma" if "t_sigma" in all_props else None)
    vel_src   = ("vel_x","vel_y","vel_z") if "vel_x" in all_props else \
                (("velocity_0","velocity_1","velocity_2") if "velocity_0" in all_props else None)
    if sigma_src:
        print(f"时域宽度：{sigma_src} → t_sigma")
    if vel_src:
        print(f"速度字段：{vel_src} → velocity_0/1/2")

    # Assemble output structured array.
    N = len(src.data)
    out_dtype = [(n, col_dtype[n]) for n in out_names]
    new_data = np.empty(N, dtype=out_dtype)
    for n in out_names:
        new_data[n] = col_data[n]

    # 静态/动态高斯分离：速度模长 < vel_threshold 的为静态高斯
    comments = [f"num_frames {num_frames}"]
    has_velocity = all(n in col_data for n in ("velocity_0", "velocity_1", "velocity_2"))
    num_static = 0

    if has_velocity and vel_threshold > 0:
        vel_sq = (new_data["velocity_0"].astype(np.float64) ** 2
                + new_data["velocity_1"].astype(np.float64) ** 2
                + new_data["velocity_2"].astype(np.float64) ** 2)
        is_static_mask = vel_sq < vel_threshold ** 2
        num_static = int(is_static_mask.sum())
        num_dynamic = N - num_static
        print(f"静态高斯（|v|<{vel_threshold}）：{num_static}  动态高斯：{num_dynamic}")

        if num_static > 0:
            static_idx  = np.where(is_static_mask)[0]
            dynamic_idx = np.where(~is_static_mask)[0]
            order = np.concatenate([static_idx, dynamic_idx])
            new_data = new_data[order]
            comments.append(f"num_static {num_static}")
    else:
        print("未启用静态分离（vel_threshold=0 或无速度属性）")

    # 动态高斯按 t_mu 排序
    if "t_mu" in out_names and num_static < N:
        dyn_chunk = new_data[num_static:].copy()
        dyn_order = np.argsort(dyn_chunk["t_mu"], kind="stable")
        new_data[num_static:] = dyn_chunk[dyn_order]
        print(f"动态高斯按 t_mu 排序（{len(dyn_chunk)} 条），支持时域窗口优化")

    new_element = PlyElement.describe(new_data, "vertex")
    PlyData([new_element], text=False, comments=comments).write(output_path)

    orig_mb = os.path.getsize(input_path) / 1024 / 1024
    slim_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n完成！")
    print(f"  总帧数：{num_frames}（已写入 PLY 头部 comment）")
    if num_static > 0:
        print(f"  静态高斯：{num_static}（预计算一次，不参与逐帧 GPU compute）")
    print(f"  原始文件：{orig_mb:.1f} MB")
    print(f"  瘦身文件：{slim_mb:.1f} MB  ({slim_mb/orig_mb*100:.1f}%)")
    print(f"  输出路径：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="4DGS PLY 属性瘦身工具（输出统一为 DT-4DGS 命名）")
    parser.add_argument("-i", "--input",  default="point_cloud.ply",      help="输入 PLY 路径（默认：point_cloud.ply）")
    parser.add_argument("-o", "--output", default="point_cloud_slim.ply",  help="输出 PLY 路径（默认：point_cloud_slim.ply）")
    parser.add_argument("-n", "--num-frames", type=int, default=None,      help="动画总帧数（不传则交互输入）")
    parser.add_argument("--vel-threshold",   type=float, default=1e-3,      help="速度模长阈值，低于此值视为静态高斯（默认 1e-3），0 禁用分离")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"错误：找不到输入文件 {args.input}")
        raise SystemExit(1)

    num_frames = args.num_frames
    if num_frames is None:
        while True:
            try:
                num_frames = int(input("请输入动画总帧数：").strip())
                if num_frames > 0:
                    break
                print("帧数必须大于 0，请重新输入。")
            except ValueError:
                print("请输入有效整数。")

    slim_ply(args.input, args.output, num_frames, args.vel_threshold)


if __name__ == "__main__":
    main()

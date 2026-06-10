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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float = 0.0   # normalized time in [0, 1] (4D extension)
    frame_idx: int = 0       # integer frame index (4D extension)

class TemporalPointCloud(NamedTuple):
    """Point cloud carrying per-point temporal metadata for TD-FastGS.

    Points are ordered static-first, then dynamic (frame-by-frame), matching the
    concatenation order expected by GaussianModel.create_from_pcd_4d.
    """
    points: np.array      # (N, 3)
    colors: np.array      # (N, 3) in [0, 1]
    normals: np.array     # (N, 3)
    timestamps: np.array  # (N,)   normalized birth time t_mu in [0, 1]
    is_static: np.array   # (N,)   bool, True for background points
    velocities: np.array = None  # (N, 3) flow-estimated initial velocity, None → zero-init

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    temporal_point_cloud: object = None  # TemporalPointCloud for 4D scenes, else None
    n_frames: int = 1                    # number of temporal frames

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

import re
import glob

def parse_frame_idx(name):
    """Extract the integer frame index from an image name.

    Looks for a `frame_<digits>` token first (the documented layout), then falls
    back to the last run of digits in the name. Returns 0 if nothing is found.
    """
    base = os.path.basename(str(name))
    m = re.search(r"frame[_-]?(\d+)", base, flags=re.IGNORECASE)
    if m is not None:
        return int(m.group(1))
    digits = re.findall(r"\d+", base)
    if digits:
        return int(digits[-1])
    return 0

def _fetch_ply_xyz_rgb(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except (ValueError, KeyError):
        colors = np.ones_like(positions) * 0.5
    return positions, colors

def load_temporal_point_cloud_pcd(scene_path, frame_to_t, flows_dir="flows"):
    """Load static + per-frame dynamic point clouds for the multi-view-video layout.

    Layout (flower300 / two):
        scene_path/static_points/*.ply           (e.g. pcd1.ply, t_mu=0)
        scene_path/dynamic_points/pcd<N>.ply      (frame N, t_mu=frame_to_t[N])

    Static points come first (t_mu=0, is_static=True), followed by dynamic points
    for each frame in ascending frame order. `frame_to_t` maps an integer frame id
    to its normalized timestamp (the same map used for the cameras), so a dynamic
    Gaussian's birth time equals its source frame's camera timestamp exactly.

    If scene_path/flows_dir exists, 3D velocities are estimated from optical flow
    and stored in the returned TemporalPointCloud.velocities field (static=0).
    """
    static_dir = os.path.join(scene_path, "static_points")
    dyn_dir = os.path.join(scene_path, "dynamic_points")

    pts_list, col_list, ts_list, static_list = [], [], [], []
    # Track per dynamic point which frame it belongs to (for flow estimation).
    frame_id_list = []

    if os.path.isdir(static_dir):
        for spath in sorted(glob.glob(os.path.join(static_dir, "*.ply")),
                            key=lambda p: parse_frame_idx(p)):
            s_pts, s_col = _fetch_ply_xyz_rgb(spath)
            if s_pts.shape[0] == 0:
                continue
            pts_list.append(s_pts)
            col_list.append(s_col)
            ts_list.append(np.zeros(s_pts.shape[0], dtype=np.float32))
            static_list.append(np.ones(s_pts.shape[0], dtype=bool))
            frame_id_list.append(np.full(s_pts.shape[0], -1, dtype=np.int32))

    if os.path.isdir(dyn_dir):
        frame_files = sorted(glob.glob(os.path.join(dyn_dir, "*.ply")),
                             key=lambda p: parse_frame_idx(p))
        for fpath in frame_files:
            fidx = parse_frame_idx(fpath)
            d_pts, d_col = _fetch_ply_xyz_rgb(fpath)
            if d_pts.shape[0] == 0:
                continue
            t = float(frame_to_t.get(fidx, 0.0))
            pts_list.append(d_pts)
            col_list.append(d_col)
            ts_list.append(np.full(d_pts.shape[0], t, dtype=np.float32))
            static_list.append(np.zeros(d_pts.shape[0], dtype=bool))
            frame_id_list.append(np.full(d_pts.shape[0], fidx, dtype=np.int32))

    if not pts_list:
        return None

    points = np.concatenate(pts_list, axis=0).astype(np.float32)
    colors = np.concatenate(col_list, axis=0).astype(np.float32)
    timestamps = np.concatenate(ts_list, axis=0).astype(np.float32)
    is_static = np.concatenate(static_list, axis=0)
    frame_ids_all = np.concatenate(frame_id_list, axis=0)
    normals = np.zeros_like(points)

    # Estimate velocities from optical flow if the flows directory exists.
    velocities = None
    flows_root = os.path.join(scene_path, flows_dir)
    if os.path.isdir(flows_root):
        dyn_mask = ~is_static
        if dyn_mask.any():
            print(f"[flow-vel] Estimating initial velocities from optical flow …")
            vel_dyn = load_flow_velocities(
                scene_path=scene_path,
                points_world=points[dyn_mask],
                frame_ids=frame_ids_all[dyn_mask],
                frame_to_t=frame_to_t,
                flows_dir=flows_dir,
            )
            velocities = np.zeros_like(points)
            velocities[dyn_mask] = vel_dyn
            v_mag = np.linalg.norm(vel_dyn, axis=1)
            print(f"[flow-vel] velocity magnitude: mean={v_mag.mean():.4f}  "
                  f"p50={np.median(v_mag):.4f}  p95={np.percentile(v_mag,95):.4f}")

    return TemporalPointCloud(points=points, colors=colors, normals=normals,
                              timestamps=timestamps, is_static=is_static,
                              velocities=velocities)

def _read_colmap_calib_full(path):
    """Like _read_colmap_calib but also returns raw intrinsic params (fx,fy,cx,cy)."""
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0", "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0", "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0", "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0", "cameras.txt"))

    cams = {}  # keyed by image name stem
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height, width = intr.height, intr.width
        R = np.transpose(qvec2rotmat(extr.qvec))   # world←cam rotation
        T = np.array(extr.tvec)                     # world-to-cam translation
        if intr.model == "SIMPLE_PINHOLE":
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        elif intr.model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
        else:
            continue  # skip unsupported models
        name = os.path.basename(extr.name).split(".")[0]
        cams[name] = {"R": R, "T": T, "fx": fx, "fy": fy,
                      "cx": cx, "cy": cy, "width": width, "height": height}
    return cams


def load_flow_velocities(scene_path, points_world, frame_ids, frame_to_t, flows_dir="flows"):
    """Estimate per-point 3D velocity from optical flow for each dynamic frame.

    For each dynamic point at frame f the function:
      1. Projects the point into every camera that has a flow file for frame f.
      2. Reads flow(u,v) at that pixel (bilinear); adds the displacement to get
         the pixel position at frame f+1.
      3. Un-projects both pixel positions at depth=1 into camera-space rays, takes
         the difference as a direction, converts to world space and normalises by
         the frame interval in normalised time (Δt).
      4. Averages the estimates from all cameras that could see the point.

    Returns float32 array (N_dynamic, 3). Points whose frame has no flow (last frame)
    or that project outside all cameras get velocity=0.

    Args:
        scene_path:    dataset root
        points_world:  (N, 3) 3-D positions of *dynamic* points only, in world space
        frame_ids:     (N,)   integer frame index for each point
        frame_to_t:    dict  frame_int → normalised timestamp
        flows_dir:     relative path inside scene_path to flow folder
    """
    calib = _read_colmap_calib_full(scene_path)
    if not calib:
        return np.zeros((len(points_world), 3), dtype=np.float32)

    flows_root = os.path.join(scene_path, flows_dir)
    N = len(points_world)
    velocities = np.zeros((N, 3), dtype=np.float32)

    unique_frames = np.unique(frame_ids)
    sorted_frames = sorted(frame_to_t.keys())
    frame_idx_map = {f: i for i, f in enumerate(sorted_frames)}

    for f in unique_frames:
        frame_flow_dir = os.path.join(flows_root, str(f))
        if not os.path.isdir(frame_flow_dir):
            continue

        # Next frame for Δt
        fi = frame_idx_map.get(int(f))
        if fi is None or fi + 1 >= len(sorted_frames):
            continue
        f_next = sorted_frames[fi + 1]
        delta_t = frame_to_t[f_next] - frame_to_t[f]
        if abs(delta_t) < 1e-8:
            continue

        mask = (frame_ids == f)
        pts = points_world[mask]   # (M, 3)
        M = pts.shape[0]
        vel_sum = np.zeros((M, 3), dtype=np.float64)
        vel_cnt = np.zeros(M, dtype=np.float32)

        for cam_name, c in calib.items():
            flow_path = os.path.join(frame_flow_dir, cam_name + ".npy")
            if not os.path.exists(flow_path):
                continue
            flow = np.load(flow_path)   # (Hf, Wf, 2)  values in flow-image coords
            Hf, Wf = flow.shape[:2]
            H_cam, W_cam = c["height"], c["width"]

            # W2C matrix
            R, T = c["R"], c["T"]
            # COLMAP convention: R is world←cam rotation (R^T is cam←world)
            # T is cam translation in world coords (applied as p_cam = R^T p_world - R^T T)
            # Actually in COLMAP: p_cam = R^T @ (p_world - T) where T = camera center in world
            # But dataset_readers stores R = transpose(qvec2rotmat) and T = tvec (not center).
            # So: p_cam = R.T @ p_world + T  (standard COLMAP w2c)
            Rc = R.T               # (3,3) cam←world rotation
            Tc = T                 # (3,)  translation part of w2c

            pts_cam = (Rc @ pts.T).T + Tc   # (M, 3)

            # Only points in front of camera
            z = pts_cam[:, 2]
            valid = z > 0.01
            if not np.any(valid):
                continue

            # Project to pixel (using full-res intrinsics)
            u0 = (pts_cam[valid, 0] / z[valid]) * c["fx"] + c["cx"]
            v0 = (pts_cam[valid, 1] / z[valid]) * c["fy"] + c["cy"]

            # Scale to flow image resolution
            scale_u = Wf / W_cam
            scale_v = Hf / H_cam
            uf = u0 * scale_u
            vf = v0 * scale_v

            # Clip to flow image bounds
            in_bounds = (uf >= 0) & (uf < Wf - 1) & (vf >= 0) & (vf < Hf - 1)
            if not np.any(in_bounds):
                continue

            valid_idx = np.where(valid)[0][in_bounds]  # indices into pts
            uf_valid = uf[in_bounds]
            vf_valid = vf[in_bounds]

            # Bilinear sample flow
            ui = uf_valid.astype(np.int32)
            vi = vf_valid.astype(np.int32)
            du = uf_valid - ui
            dv = vf_valid - vi
            # flow is (Hf, Wf, 2): [delta_u_in_flow_coords, delta_v_in_flow_coords]
            f00 = flow[vi,   ui  ]   # (K, 2)
            f10 = flow[vi+1, ui  ]
            f01 = flow[vi,   ui+1]
            f11 = flow[vi+1, ui+1]
            flow_uv = (f00 * (1-du[:,None]) * (1-dv[:,None])
                     + f01 * (1-dv[:,None]) * du[:,None]
                     + f10 * dv[:,None]     * (1-du[:,None])
                     + f11 * dv[:,None]     * du[:,None])  # (K, 2)

            # Convert flow from flow-image coords to full-res pixel coords
            flow_u_px = flow_uv[:, 0] / scale_u
            flow_v_px = flow_uv[:, 1] / scale_v

            # Pixel coords at frame f+1
            u1 = u0[in_bounds] + flow_u_px
            v1 = v0[in_bounds] + flow_v_px

            # Unproject both pixels at depth=1 → direction in camera space
            inv_fx, inv_fy = 1.0 / c["fx"], 1.0 / c["fy"]
            dir0 = np.stack([(u0[in_bounds] - c["cx"]) * inv_fx,
                             (v0[in_bounds] - c["cy"]) * inv_fy,
                             np.ones(len(u1))], axis=1)   # (K, 3)
            dir1 = np.stack([(u1 - c["cx"]) * inv_fx,
                             (v1 - c["cy"]) * inv_fy,
                             np.ones(len(u1))], axis=1)   # (K, 3)

            # Scale directions by actual depth so displacement is metric
            depth = z[valid][in_bounds]
            dir0 *= depth[:, None]
            dir1 *= depth[:, None]

            # Displacement in camera space → world space (rotation only, no translation)
            Rw = Rc.T    # world←cam
            disp_world = (Rw @ (dir1 - dir0).T).T   # (K, 3)

            vel_world = disp_world / delta_t
            vel_sum[valid_idx] += vel_world
            vel_cnt[valid_idx] += 1

        has_est = vel_cnt > 0
        velocities[mask] = np.where(has_est[:, None],
                                    (vel_sum / np.maximum(vel_cnt[:, None], 1)).astype(np.float32),
                                    0.0)
        n_est = int(has_est.sum())
        print(f"[flow-vel] frame {f}: {M} pts, {n_est} got flow estimate "
              f"({M - n_est} zero-init)")

    return velocities


def _read_colmap_calib(path):
    """Read COLMAP camera calibration without opening any images.

    Returns a list of dicts {uid, R, T, FovX, FovY, width, height, name} ordered
    by image name. `name` is the COLMAP image NAME stem (here a camera id like
    "1", "2", ...). Supports both binary and text COLMAP exports.
    """
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0", "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0", "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0", "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0", "cameras.txt"))

    cams = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height, width = intr.height, intr.width
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model == "SIMPLE_PINHOLE":
            fx = intr.params[0]
            FovY = focal2fov(fx, height)
            FovX = focal2fov(fx, width)
        elif intr.model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)
        else:
            assert False, ("Colmap camera model not handled: only undistorted "
                           "datasets (PINHOLE or SIMPLE_PINHOLE) supported!")
        cams.append({"uid": intr.id, "R": R, "T": T, "FovX": FovX, "FovY": FovY,
                     "width": width, "height": height,
                     "name": os.path.basename(extr.name).split(".")[0]})
    cams.sort(key=lambda c: c["name"])
    return cams


def readColmap4DSceneInfo(path, images, eval, n_frames=-1, llffhold=8):
    """4DGS multi-view-video reader (flower300 layout).

    `sparse/0` calibrates a fixed set of cameras (the COLMAP image names are CAMERA
    ids, not frames). The actual frames live as folders under `images/<frame>/images/`,
    each holding one image per camera. Training views are therefore the cross
    product (camera x frame); a Camera is emitted per (camera, frame) pair with the
    frame's normalized timestamp. Images are NOT opened here (lazy loading): the
    CameraInfo carries `image=None` and the path, and dims come from the COLMAP
    intrinsics. The decoupled static/dynamic .ply clouds provide 4D initialization.
    """
    reading_dir = "images" if images is None else images
    images_root = os.path.join(path, reading_dir)

    calib = _read_colmap_calib(path)

    # Discover integer-named frame folders under images/.
    frames = []
    if os.path.isdir(images_root):
        for entry in os.listdir(images_root):
            if entry.isdigit() and os.path.isdir(os.path.join(images_root, entry)):
                frames.append(int(entry))
    frames.sort()
    if not frames:
        frames = [0]
    fmin, fmax = frames[0], frames[-1]
    span = float(fmax - fmin) if fmax > fmin else 1.0
    frame_to_t = {f: (float(f - fmin) / span) for f in frames}

    if n_frames is None or n_frames <= 0:
        n_frames = len(frames)

    # Cross product: one CameraInfo per (frame, camera), lazily loaded.
    cam_infos = []
    uid = 0
    missing = 0
    for f in frames:
        frame_dir = os.path.join(images_root, str(f), "images")
        t = frame_to_t[f]
        for c in calib:
            img_path = os.path.join(frame_dir, c["name"] + ".png")
            if not os.path.exists(img_path):
                missing += 1
                continue
            cam_infos.append(CameraInfo(
                uid=uid, R=c["R"], T=c["T"], FovY=c["FovY"], FovX=c["FovX"],
                image=None, image_path=img_path,
                image_name="f{}_c{}".format(f, c["name"]),
                width=c["width"], height=c["height"],
                timestamp=t, frame_idx=f))
            uid += 1
    if missing:
        print("[TD-FastGS] Warning: {} (camera, frame) images were missing and skipped.".format(missing))
    print("[TD-FastGS] Built {} cameras across {} frames ({} calibrated cams).".format(
        len(cam_infos), len(frames), len(calib)))

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    temporal_pcd = load_temporal_point_cloud_pcd(path, frame_to_t)
    if temporal_pcd is not None:
        pcd = BasicPointCloud(points=temporal_pcd.points,
                              colors=temporal_pcd.colors,
                              normals=temporal_pcd.normals)
    else:
        pcd = None
    # The decoupled clouds are the only init source (points3D is empty here); no
    # input.ply round-trip, so report no ply_path.
    ply_path = None

    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cam_infos,
                     test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization,
                     ply_path=ply_path,
                     temporal_point_cloud=temporal_pcd,
                     n_frames=n_frames)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Colmap4D": readColmap4DSceneInfo
}
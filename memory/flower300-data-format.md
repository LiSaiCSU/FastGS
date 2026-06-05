---
name: flower300-data-format
description: Layout of the user's flower300 multi-view-video 4D dataset
metadata:
  type: project
---
The user's TD-FastGS data (`flower300/`, also the target real-data format) is a multi-view video, NOT the `points3D.ply`/`frame_*.ply` layout assumed by the original prompt.md spec:

- `sparse/0/{cameras,images,points3D}.txt` — COLMAP calibration for 36 fixed cameras (PINHOLE, 3839x2159). `images.txt` names them `1.png`..`36.png` (these are CAMERA ids, not frames). `points3D.txt` is EMPTY — no COLMAP point cloud; init comes only from the PLYs below.
- `images/<frame>/images/<cam>.png` — frames 1..300, each folder has 36 cam PNGs (some frames also have redundant `.jpg` duplicates that must be ignored). Training cameras = cross product 36 cams x 300 frames = 10,800 images.
- `static_points/pcd1.ply` — ~17k static background pts (t_mu=0).
- `dynamic_points/pcd<frame>.ply` — frames 1..300, ~2300 pts each, born at that frame's time.

Timestamp normalization MUST be identical for cameras and dynamic PLYs: `t = (frame_id - fmin)/(fmax - fmin)`, fmin=1 fmax=300. Static => t=0.

10,800 full-res images cannot be eagerly loaded to GPU — Camera needs lazy disk loading with a bounded LRU CPU cache. See [[fast4dgs-td-implementation]].

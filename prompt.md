# TD-FastGS 实现提示词
> 提供给 Claude Opus 4.8 (claude-opus-4-8) 的完整实现指导文档
> 任务：以 FastGS 为基础代码，移植 TD-4DGS 的时域机制，实现高效的 4D Gaussian Splatting

---

## 你的任务

你是一名计算机视觉领域的资深工程师，需要将一套名为 **TD-4DGS** 的时域扩展机制移植到 **FastGS** 代码库中，实现 **TD-FastGS**：一个在保持 FastGS 训练加速优势的同时，支持动态场景重建的 4D Gaussian Splatting 系统。

请严格按照本文档逐步实现，不要跳过任何小节，不要做出文档未要求的"顺手优化"。每完成一个模块后，写出该模块的单元测试代码（不需要运行，只需写出测试逻辑）。

---

## 背景知识

### FastGS 的核心机制

FastGS 在原版 3DGS 的基础上做了三项改进：

**1. 多视图一致性致密化（VCD）**：不再用图像空间梯度幅值判断是否致密化，而是随机采样 K 个视角，对每个视角生成逐像素 L1 误差图（min-max 归一化），提取高误差掩码，统计每个高斯在其 2D footprint 内跨多视角的高误差像素均值作为致密化重要性分数：

$$s^i_d = \frac{1}{K} \sum_{j=1}^{K} \sum_{p \in \Omega_i} \mathbb{I}\left(M^j_{mask}(p) = 1\right)$$

仅当 $s^i_d > \tau_d$（默认=5）时才执行 clone/split。

**2. 多视图一致性剪枝（VCP）**：结合逐视角光度损失 $E^j_{photo}$，计算剪枝分数：

$$s^i_p = \mathcal{N}\left(\sum_{j=1}^{K} \left(\sum_{p \in \Omega_i} \mathbb{I}\left(M^j_{mask}(p) = 1\right)\right) \cdot E^j_{photo}\right)$$

当 $s^i_p > \tau_p$（默认=0.9）时删除该高斯。

**3. Compact Box（CB）**：在光栅化预处理阶段，用 Mahalanobis 距离阈值代替 3-sigma 规则，进一步减少 Gaussian-tile 对：

$$(\mathbf{p} - \mu_{i_{2D}}) \Sigma^{-1}_{i_{2D}} (\mathbf{p} - \mu_{i_{2D}})^T \leq \beta \left(2\ln\frac{\sigma_i}{\tau_\alpha}\right)$$

FastGS 基于 `3DGS-accel`（集成了 Taming-3DGS 的 per-splat 并行反传和 SH 加速），默认 30K 迭代，K=10，λ=0.2，NVIDIA RTX 4090 上约 100 秒完成训练。

---

### TD-4DGS 的时域机制

TD-4DGS 在每个高斯基元上附加 **5 个显式时域标量**：

| 属性 | 符号 | 可学习 | 语义 |
|------|------|--------|------|
| 出生时间 | $t_\mu$ | **否** | 锚定于 SfM 帧索引，归一化到 [0,1] |
| 生命半径（log空间） | $\sigma_{t,raw}$ | 是 | $\sigma_t = e^{\sigma_{t,raw}}$ |
| 运动速度 | $\mathbf{v} \in \mathbb{R}^3$ | 是（动态点），锁死（静态点） |

时空变换：
$$\mathbf{x}'(t) = \mathbf{x}_0 + \mathbf{v} \cdot (t - t_\mu)$$
$$\alpha'_i(t) = \alpha_i \cdot \underbrace{\exp\left(-\frac{(t - t_\mu^{(i)})^2}{2\sigma_t^{(i)2} + \epsilon}\right)}_{w_t^{(i)}(t)}$$

因果存活条件：
$$\text{alive}(i, t) = \mathbb{1}\left[t_\mu^{(i)} \leq t\right] \wedge \mathbb{1}\left[\alpha'_i(t) > \tau_{alive}\right], \quad \tau_{alive} = 0.005$$

每帧仅光栅化满足 alive 条件的稀疏子集。

---

## 数据格式约定

输入数据结构：
```
dataset/
├── static_points/          # 背景 SfM 点云（单帧或多帧均值）
│   └── points3D.ply
├── dynamic_points/         # 逐帧前景 SfM 点云
│   ├── frame_0000.ply
│   ├── frame_0001.ply
│   └── ...
├── images/
│   ├── cam_00/
│   │   ├── frame_0000.jpg
│   │   └── ...
│   └── cam_01/
│       └── ...
└── sparse/                 # COLMAP 标定结果（cameras.bin, images.bin）
```

- $N_{cam}$：固定视角相机数（如 36）
- $N_{frame}$：时序帧数（如 80-150）
- 每帧每视角一张图像，时间戳归一化为 $t \in [0, 1]$

---

## 文件修改清单

需要修改的文件（以 FastGS 代码库为基础）：

| 文件 | 修改性质 |
|------|---------|
| `scene/gaussian_model.py` | **核心扩展**：时域属性注册、初始化、PLY 序列化、时域感知 ADC |
| `gaussian_renderer/__init__.py` | **渲染扩展**：时空变换、因果剪枝、alive_mask 回填 |
| `train.py` | **训练逻辑**：梯度闸门、静态硬拉回、解耦 opacity reset、3 阶段采样、时域感知 VCD/VCP |
| `scene/__init__.py` | 场景管理：4DGS 自动检测、解耦点云加载 |
| `scene/dataset_readers.py` | 数据读取：多帧场景读取器、时间戳赋值 |
| `scene/cameras.py` | 相机对象：timestamp 属性 |

---

## 模块一：高斯模型时域扩展（`gaussian_model.py`）

### 1.1 时域属性注册

在 `GaussianModel.__init__` 中新增以下张量，与现有属性并列注册到 optimizer 参数组：

```python
# 时域属性（仅动态点有效，静态点锁死）
self._t_mu = torch.empty(0)           # shape (N,)，出生时间，不可学习
self._sigma_t_raw = torch.empty(0)    # shape (N,)，生命半径 log 空间，可学习
self._velocity = torch.empty(0)       # shape (N, 3)，运动速度，可学习（动态点）
self.is_static = torch.empty(0, dtype=torch.bool)  # shape (N,)，身份掩码，不进优化器
```

重要：`_t_mu` 和 `is_static` **不加入 optimizer**，仅作为常驻属性。`_sigma_t_raw` 和 `_velocity` 加入 optimizer，但静态点的对应行梯度在每步后被硬清零。

### 1.2 初始化策略

静态点初始化：
```python
t_mu_static = torch.zeros(N_static)
sigma_t_raw_static = torch.full((N_static,), math.log(1000.0))  # sigma=1000，全时域可见
velocity_static = torch.zeros(N_static, 3)
is_static_flags = torch.ones(N_static, dtype=torch.bool)
```

动态点初始化（逐帧 SfM 点云加载后）：
```python
# timestamp 为该帧归一化时间戳，N_frames 为总帧数
t_mu_dynamic = torch.full((N_dynamic,), timestamp)  # 锚定到帧索引
sigma_t_raw_dynamic = torch.full(
    (N_dynamic,), math.log(2.5 / N_frames)
)  # 初始覆盖约 2.5 帧
velocity_dynamic = torch.zeros(N_dynamic, 3)
is_static_flags = torch.zeros(N_dynamic, dtype=torch.bool)
```

拼接顺序：静态点在前，动态点在后（便于后续 masking）。

### 1.3 时域权重计算（核心辅助函数）

```python
def compute_temporal_weight(self, t: float) -> torch.Tensor:
    """
    计算所有高斯在时刻 t 的时域活跃权重 w_t^(i)(t)
    返回 shape (N,) 的权重张量，静态点权重恒为 1.0
    """
    sigma_t = torch.exp(self._sigma_t_raw)  # (N,)
    dt = t - self._t_mu                     # (N,)
    w_t = torch.exp(-dt ** 2 / (2 * sigma_t ** 2 + 1e-8))  # (N,)
    # 静态点恒为 1.0
    w_t[self.is_static] = 1.0
    return w_t
```

### 1.4 时域感知 VCD（替换原版 `densify_and_clone` / `densify_and_split`）

**修改要点**：在 FastGS 原始 VCD 的多视图分数计算中，引入时域权重：

```python
def compute_vcd_score_4d(self, viewspace_points_list, visibility_filter_list,
                          rendered_images, gt_images, timestamps, tau=0.5):
    """
    时域感知 VCD 分数计算。
    
    参数：
        viewspace_points_list: 每个采样视角的 2D 投影点列表
        visibility_filter_list: 每个视角的可见性掩码列表
        rendered_images: list of (3, H, W)，K 个采样视角的渲染结果
        gt_images: list of (3, H, W)，对应的 GT 图像
        timestamps: list of float，每个采样视角的时间戳
        tau: 高误差像素阈值
    
    返回：
        scores: shape (N,)，时域加权 VCD 分数
    """
    N = self._xyz.shape[0]
    scores = torch.zeros(N, device="cuda")
    
    for j, (render_j, gt_j, t_j, pts_j, vis_j) in enumerate(
        zip(rendered_images, gt_images, timestamps,
            viewspace_points_list, visibility_filter_list)
    ):
        # 1. 计算逐像素 L1 误差图并归一化
        err_map = (render_j - gt_j).abs().mean(dim=0)  # (H, W)
        err_map_norm = (err_map - err_map.min()) / (err_map.max() - err_map.min() + 1e-8)
        mask_j = (err_map_norm > tau).float()  # (H, W)
        
        # 2. 计算当前视角时间戳下各高斯的时域权重
        w_t = self.compute_temporal_weight(t_j)  # (N,)
        
        # 3. 从渲染器前向传播中获取高误差像素计数
        # （FastGS 原版在 render 前向传播中直接统计 2D footprint 内的高误差像素数）
        # 这里用 w_t 对每个高斯的像素计数进行加权
        pixel_count_j = self._get_footprint_error_count(pts_j, vis_j, mask_j)  # (N,)
        
        scores = scores + w_t * pixel_count_j
    
    return scores / len(timestamps)

def _get_footprint_error_count(self, viewspace_pts, visibility, error_mask):
    """
    统计每个可见高斯在其 2D footprint 内的高误差像素数。
    复用 FastGS 渲染器前向传播中已实现的统计逻辑。
    不可见的高斯返回 0。
    """
    N = self._xyz.shape[0]
    counts = torch.zeros(N, device="cuda")
    # 实现参考 FastGS 原版 render 中的 footprint 统计
    # ...（保持与 FastGS 原版 VCD 一致的实现方式，只是在外部加 w_t 权重）
    return counts
```

**致密化条件修改**：

```python
def densify_and_prune_4d(self, vcd_scores, vcp_scores, tau_d=5.0, tau_p=0.9,
                          min_opacity=0.005, max_screen_size=None):
    """
    时域感知 ADC 主函数，替换 FastGS 原版 densify_and_prune。
    
    VCD（致密化）：仅对 w_t 活跃期内且 VCD 分数超过阈值的高斯执行 clone/split
    VCP（剪枝）：分静态/动态两套逻辑
    """
    # --- VCD: 致密化 ---
    # 基础条件：VCD 分数超阈值
    densify_mask = vcd_scores > tau_d
    
    # 动态点额外条件：仅在活跃期（w_t > 0.2）内允许致密化
    # （通过采样当前 batch 时间戳的 w_t 均值估计）
    dynamic_active = (~self.is_static) & (self._current_wt_mean > 0.2)
    static_mask = self.is_static
    densify_mask = densify_mask & (static_mask | dynamic_active)
    
    # 执行 clone/split（与 FastGS 原版逻辑相同，子代继承 is_static 和时域属性）
    self._clone_with_temporal(densify_mask & (grad < threshold))
    self._split_with_temporal(densify_mask & (grad >= threshold))
    
    # --- VCP: 剪枝 ---
    # 静态点：原版 VCP 逻辑
    static_prune = self.is_static & (vcp_scores > tau_p)
    
    # 动态点：Credit-Assigned 联合剪枝
    # 仅在活跃期（w_t > 0.2）内判断，避免非活跃帧的正常动态点被误杀
    dynamic_active_prune = (~self.is_static) & (self._current_wt_mean > 0.2)
    dynamic_prune = dynamic_active_prune & (vcp_scores > tau_p)
    
    prune_mask = static_prune | dynamic_prune
    
    # 附加：过小/过透明点
    opacity_prune = (self.get_opacity.squeeze() < min_opacity)
    if max_screen_size is not None:
        size_prune = self.get_scaling.max(dim=1).values > max_screen_size
        prune_mask = prune_mask | opacity_prune | size_prune
    else:
        prune_mask = prune_mask | opacity_prune
    
    self.prune_points(prune_mask)
```

### 1.5 clone/split 中的时域属性继承

```python
def _clone_with_temporal(self, mask):
    """Clone 时，子代继承父代所有属性，包括时域属性"""
    # 克隆所有属性（参考 FastGS 原版 clone 实现）
    new_xyz = self._xyz[mask]
    new_t_mu = self._t_mu[mask]
    new_sigma_t_raw = self._sigma_t_raw[mask]
    new_velocity = self._velocity[mask]
    new_is_static = self.is_static[mask]
    # ... 其他属性同原版
    
    self._append_gaussians(new_xyz, new_t_mu, new_sigma_t_raw,
                           new_velocity, new_is_static, ...)

def _split_with_temporal(self, mask, N_splits=2):
    """Split 时，子代继承父代时域属性（位置扰动，时域参数不变）"""
    # 子代时域属性 = 父代时域属性（直接复制）
    # 子代位置 = 父代位置 + 沿主轴方向的随机扰动
    # 子代 scale 缩小（原版行为）
    # 注意：静态点子代的 velocity 和 sigma_t_raw 需要在 post-optim 中硬拉回
    pass
```

### 1.6 Credit-Assigned 剪枝（VCP 的动态点条件）

在 VCP 分数计算之外，以下情况的动态点需额外保护（不被 VCP 删除）：
- `w_t < 0.2`：当前非活跃期，即使 VCP 分数高也不剪枝（可能下一帧才是活跃期）
- `alpha * w_t > 0.005`：在活跃期内仍有足够不透明度

---

## 模块二：渲染管线时域扩展（`gaussian_renderer/__init__.py`）

### 2.1 渲染主函数修改

在调用 CUDA 光栅化器之前，插入时空变换和因果剪枝：

```python
def render_4d(viewpoint_camera, pc: GaussianModel, pipe, bg_color,
              scaling_modifier=1.0, override_color=None):
    """
    4D 渲染主函数。
    
    关键顺序（不可更改）：
    1. 时空变换：将高斯中心平移到当前帧位置
    2. 因果剪枝：生成 alive_mask 稀疏子集
    3. Compact Box 计算：在 alive 子集上计算 CB（重要：必须在此顺序）
    4. 光栅化：对 alive 子集进行 tile-based rasterization
    5. alive_mask 回填：将稀疏子集统计量映射回全量尺寸
    """
    t = viewpoint_camera.timestamp  # float in [0, 1]
    
    # Step 1: 时空变换
    w_t = pc.compute_temporal_weight(t)  # (N,)
    
    # 位置变换：x' = x0 + v * (t - t_mu)
    dt = t - pc._t_mu  # (N,)
    xyz_transformed = pc.get_xyz + pc._velocity * dt.unsqueeze(-1)  # (N, 3)
    
    # 时域有效不透明度：alpha' = alpha * w_t
    opacity = pc.get_opacity.squeeze(-1) * w_t  # (N,)
    
    # Step 2: 因果剪枝（alive_mask）
    # 条件1：t_mu <= t（因果律，高斯未"出生"则不渲染）
    causal_mask = pc._t_mu <= t + 1e-6  # (N,) bool
    # 条件2：alpha' > tau_alive（时域活跃度不透明度阈值）
    alive_mask = causal_mask & (opacity > 0.005)  # (N,) bool
    
    # Step 3 & 4: 在 alive 子集上执行 FastGS 原版渲染流程（含 CB）
    # 提取稀疏子集
    alive_idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)
    xyz_alive = xyz_transformed[alive_idx]
    opacity_alive = opacity[alive_idx]
    # ... 提取其他属性子集
    
    # 调用 FastGS 的 CB 光栅化器（传入变换后的位置）
    rendered_image, radii_sparse, viewspace_pts_sparse = rasterize_cb(
        xyz_alive, opacity_alive, ...
    )
    
    # Step 5: 将稀疏子集的 radii 和 viewspace_pts 回填到全量尺寸
    radii_full = torch.zeros(pc._xyz.shape[0], device="cuda")
    viewspace_pts_full = torch.zeros(pc._xyz.shape[0], 2, device="cuda")
    radii_full[alive_idx] = radii_sparse
    viewspace_pts_full[alive_idx] = viewspace_pts_sparse
    
    return {
        "render": rendered_image,
        "viewspace_points": viewspace_pts_full,   # 全量尺寸，与 FastGS VCD 统计兼容
        "visibility_filter": alive_mask,           # 全量尺寸
        "radii": radii_full,
        "w_t": w_t,  # 返回供 train.py 使用
    }
```

### 2.2 alive_mask 的双层回填

回填时注意：FastGS 的 VCD 统计是在 **全量高斯的 viewspace_points 上运行**的（用 radii 判断 footprint）。回填必须保持 shape 和 dtype 与原版一致，否则 VCD 统计的 footprint 计算会出错。

具体地：
- `radii_full[~alive_mask] = 0`：未激活高斯的 radii 置 0，使其 footprint 为空集
- `viewspace_pts_full`：需要携带 `requires_grad=True`，因为 FastGS 通过 `viewspace_points.grad` 统计致密化梯度

---

## 模块三：训练循环修改（`train.py`）

### 3.1 三级梯度闸门

在每次 `loss.backward()` 后、`optimizer.step()` 前插入：

```python
def apply_gradient_gating(gaussians: GaussianModel, t_current: float):
    """
    三级梯度闸门：
    - 静态点：v 和 sigma_t_raw 的梯度强制清零
    - 动态点（当前帧，即 t 接近 t_mu）：所有参数梯度放行
    - 动态点（其他帧）：xyz/f_dc/f_rest/scaling/rotation 梯度清零，
                        只允许 opacity/velocity/sigma_t_raw 梯度通过
    
    注意：此函数中"当前帧"的判断使用 w_t > 0.5 作为阈值（在 ~1σ 生命周期核心内）
    """
    w_t = gaussians.compute_temporal_weight(t_current)  # (N,)
    
    is_static = gaussians.is_static                      # (N,) bool
    is_dynamic_current = (~is_static) & (w_t > 0.5)    # 动态且在当前帧活跃窗口内
    is_dynamic_other = (~is_static) & (w_t <= 0.5)     # 动态但不在当前帧
    
    # 静态点：锁死速度和时域参数
    for param_name in ['_velocity', '_sigma_t_raw']:
        param = getattr(gaussians, param_name)
        if param.grad is not None:
            param.grad[is_static] = 0.0
    
    # 动态点（其他帧）：锁死几何参数，只允许 opacity/velocity/sigma_t_raw 更新
    geo_params = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation']
    for param_name in geo_params:
        param = getattr(gaussians, param_name)
        if param.grad is not None:
            param.grad[is_dynamic_other] = 0.0
    
    # 注意：opacity 梯度对所有动态点（包括 other 帧）都放行
    # 这是允许 opacity 跨时间轴统筹优化的关键设计
```

### 3.2 静态点硬拉回（每步 optimizer.step() 后执行）

```python
def enforce_static_constraints(gaussians: GaussianModel):
    """
    物理硬拉回，对抗 Adam 动量残余。
    必须在 optimizer.step() 之后立即调用。
    """
    with torch.no_grad():
        static_mask = gaussians.is_static
        
        # 速度强制归零
        gaussians._velocity.data[static_mask] = 0.0
        
        # 生命半径强制设为全时域可见（log(1000)）
        gaussians._sigma_t_raw.data[static_mask] = math.log(1000.0)
        
        # 出生时间强制为 0
        gaussians._t_mu.data[static_mask] = 0.0
        
        # 同步清除对应的 Adam 动量（防止动量把参数拉回去）
        for group in gaussians.optimizer.param_groups:
            for p in group['params']:
                if p is gaussians._velocity or p is gaussians._sigma_t_raw:
                    state = gaussians.optimizer.state[p]
                    if 'exp_avg' in state:
                        state['exp_avg'][static_mask] = 0.0
                        state['exp_avg_sq'][static_mask] = 0.0
```

### 3.3 解耦 opacity 重置策略

```python
def reset_opacity_decoupled(gaussians: GaussianModel, reset_value: float = 0.01):
    """
    分治重置：
    - 静态点：重置到 reset_value（原版行为）
    - 动态点：不重置（保护动态前景的 opacity 状态）
    
    同时同步 Adam 状态，防止重置失效。
    """
    with torch.no_grad():
        static_mask = gaussians.is_static
        
        # 仅对静态点执行 opacity 重置
        opacities_new = gaussians.get_opacity.clone()
        opacities_new[static_mask] = inverse_sigmoid(
            torch.ones(static_mask.sum(), 1, device="cuda") * reset_value
        )
        
        # 通过 replace_tensor_to_optimizer 同步更新 Adam 状态
        gaussians.replace_tensor_to_optimizer(opacities_new, "opacity")
```

### 3.4 时序感知相机采样策略（3 阶段）

```python
def sample_camera_4d(train_cameras: List, iteration: int,
                     N_frames: int, N_cam_per_frame: int) -> Camera:
    """
    3 阶段采样策略：
    
    阶段1（iter <= 3000）：静态强化期
        仅从第 0 帧的 N_cam 个视角中采样，优先收敛静态背景基座
    
    阶段2（3000 < iter <= 10000）：时序滑窗期
        在时间轴上随机选一个起始帧，取相邻 W=4 帧内的视角
        为速度 v 提供连续帧对比梯度（速度的唯一有效监督信号）
    
    阶段3（iter > 10000）：全局随机期
        全部训练相机无放回随机采样，保证全时域全视角覆盖
    """
    if iteration <= 3000:
        # 阶段1：仅第 0 帧
        frame_0_cameras = [c for c in train_cameras if c.frame_idx == 0]
        return random.choice(frame_0_cameras)
    
    elif iteration <= 10000:
        # 阶段2：时序滑窗
        window_size = 4
        start_frame = random.randint(0, N_frames - window_size)
        window_cameras = [
            c for c in train_cameras
            if start_frame <= c.frame_idx < start_frame + window_size
        ]
        return random.choice(window_cameras)
    
    else:
        # 阶段3：全局随机
        return random.choice(train_cameras)
```

### 3.5 时域感知 VCD/VCP 集成到训练主循环

FastGS 原版的 VCD/VCP 调用逻辑是：在每次 densification 时重新渲染 K 个采样视角，获取误差图和光度损失，计算分数。

在 4D 版本中，修改采样视角的方式：

```python
def sample_views_for_vcd_vcp(
    train_cameras: List,
    K: int = 10,
    iteration: int = 0
) -> List[Camera]:
    """
    为 VCD/VCP 采样 K 个视角。
    
    与相机采样策略对齐：
    - iter <= 3000：只从第 0 帧采样（保证静态场景一致性）
    - iter > 3000：全局随机采样（覆盖时域，但注意时域权重会自动过滤非活跃视角的贡献）
    
    返回：K 个 Camera 对象，包含 timestamp 属性
    """
    if iteration <= 3000:
        pool = [c for c in train_cameras if c.frame_idx == 0]
    else:
        pool = train_cameras
    
    return random.sample(pool, min(K, len(pool)))
```

在计算 VCD/VCP 分数时，将每个采样视角的时间戳 `t_j` 传入，使 `compute_temporal_weight(t_j)` 自动对非活跃高斯的贡献置零。

### 3.6 训练主循环时间线

```
iter     0 ─────────────────────────────────────────────── 30000
         │
    500  ├─ 致密化开始（VCD + VCP，每 500 轮）
   2000  ├─ SH → 1 阶
   3000  ├─ 首次 opacity reset（仅静态点）
         │  结束静态专属采样 → 进入时序滑窗期
   6000  ├─ opacity reset（仅静态点）+ SH → 2 阶
  10000  ├─ opacity reset（仅静态点）+ SH → 3 阶（满阶）
         │  进入全局随机期
  12000  ├─ opacity reset（仅静态点）
  15000  ├─ opacity reset（仅静态点）+ 致密化结束
         │  之后 VCP 每 3000 轮执行一次（仅剪枝，不致密化）
  30000  └─ 训练结束，保存 PLY 序列
```

---

## 模块四：场景与数据读取（`scene/` 目录）

### 4.1 相机对象扩展（`cameras.py`）

在 `Camera` 类中添加：

```python
class Camera:
    def __init__(self, ..., timestamp: float = 0.0, frame_idx: int = 0):
        # 现有属性...
        self.timestamp = timestamp   # float in [0, 1]，归一化时间戳
        self.frame_idx = frame_idx   # int，帧序号
        
        # 延迟图像加载：仅记录路径，不解码像素
        # self._image_path = image_path
        # self._image = None  # 懒加载
    
    @property
    def original_image(self):
        # 懒加载实现（可选，用于大规模数据集）
        if self._image is None:
            self._image = load_image_to_tensor(self._image_path)
        return self._image
```

### 4.2 场景读取器（`dataset_readers.py`）

```python
def read_4dgs_scene(scene_path: str, N_frames: int) -> Tuple[List, PointCloud]:
    """
    读取 4DGS 场景数据：
    
    1. 读取 COLMAP 标定结果（cameras.bin + images.bin）
    2. 构建 Camera 对象列表，分配 timestamp = frame_idx / (N_frames - 1)
    3. 读取 static_points/points3D.ply 作为背景点云
    4. 逐帧读取 dynamic_points/frame_XXXX.ply 作为前景点云
    5. 返回 (camera_list, StaticPointCloud, DynamicPointCloudList)
    
    前景点云打包方式：DynamicPointCloudList[i] 是第 i 帧的点云，
    每个点带 timestamp = i / (N_frames - 1)，用于初始化 t_mu
    """
    pass
```

---

## 模块五：损失函数

```python
# 总损失（与 TD-4DGS 保持一致）
loss = (1 - lambda_s) * L1 + lambda_s * (1 - SSIM) + lambda_v * L_smooth

# 速度平滑正则（仅动态点）
def compute_velocity_smoothness_loss(gaussians: GaussianModel, K_pairs: int = 4096):
    """
    随机采样 K_pairs 对动态点，计算空间高斯核加权的速度一致性损失。
    
    L_smooth = (1/K) * sum_k w_k * ||v_{a_k} - v_{b_k}||^2
    w_k = exp(-||x_{a_k} - x_{b_k}||^2 / (2 * s_bar^2))
    
    其中 s_bar^2 是局部空间尺度（用 KNN 距离估计）。
    """
    dynamic_idx = (~gaussians.is_static).nonzero(as_tuple=False).squeeze(-1)
    if dynamic_idx.shape[0] < 2:
        return torch.tensor(0.0, device="cuda")
    
    # 随机采样点对
    idx_a = torch.randint(0, dynamic_idx.shape[0], (K_pairs,))
    idx_b = torch.randint(0, dynamic_idx.shape[0], (K_pairs,))
    
    pos_a = gaussians.get_xyz[dynamic_idx[idx_a]]   # (K, 3)
    pos_b = gaussians.get_xyz[dynamic_idx[idx_b]]   # (K, 3)
    vel_a = gaussians._velocity[dynamic_idx[idx_a]] # (K, 3)
    vel_b = gaussians._velocity[dynamic_idx[idx_b]] # (K, 3)
    
    dist_sq = ((pos_a - pos_b) ** 2).sum(-1)        # (K,)
    s_bar_sq = dist_sq.mean().detach() + 1e-8       # 全局尺度估计（简化版）
    
    w = torch.exp(-dist_sq / (2 * s_bar_sq))        # (K,)
    loss = (w * ((vel_a - vel_b) ** 2).sum(-1)).mean()
    
    return loss

# 可选：深度正则（指数衰减权重）
depth_weight = 1.0 * math.exp(-iteration / 5000.0)  # 从 1.0 衰减到 ~0.007 at 30K
loss = loss + depth_weight * L_depth  # 仅在有深度图时启用
```

---

## 模块六：动态 Scale 约束（可选，谨慎使用）

如果发现动态高斯出现"膨胀偷懒"（少量大球代替大量小球），可以在 optimizer.step() 后施加软惩罚，**不建议硬钳位**（硬钳位会破坏 Adam 动量）：

```python
# 软惩罚替代硬钳位
scale_limit = math.log(0.05 * scene_extent)
dynamic_mask = ~gaussians.is_static
scale_excess = (gaussians._scaling[dynamic_mask] - scale_limit).clamp(min=0)
scale_penalty = scale_excess.pow(2).mean()
loss = loss + 0.001 * scale_penalty  # 权重可调
```

---

## 关键注意事项（必读）

### ⚠️ 注意1：alive_mask 与 CB 的执行顺序
**CB 必须在 alive_mask 过滤之后执行**。如果先计算 CB 再过滤 alive 点，会对不该渲染的高斯浪费 tile pair 计算。正确顺序：提取 alive 子集 → 对子集执行 CB → 光栅化。

### ⚠️ 注意2：VCD 时域权重不能在 `torch.no_grad()` 块中计算
VCD/VCP 分数计算时，`compute_temporal_weight` 必须在计算图中（需要梯度回传到 `sigma_t_raw`）。不要在 `with torch.no_grad():` 块中调用。

### ⚠️ 注意3：FastGS 的 τ_d 阈值需为动态点降低
FastGS 原版 τ_d=5 是针对静态场景调优的，动态场景中每个高斯覆盖的视角更少（因为时域滤波），导致致密化分数天然偏低。建议对动态点使用 τ_d_dynamic = τ_d_static × 0.5 = 2.5。

```python
# 在致密化条件中：
tau_d_effective = torch.where(
    gaussians.is_static,
    torch.tensor(5.0),
    torch.tensor(2.5)   # 动态点阈值减半
)
densify_mask = vcd_scores > tau_d_effective
```

### ⚠️ 注意4：时域权重的梯度截断位置
在渲染时，`opacity = base_opacity * w_t` 中，`w_t` 对 `sigma_t_raw` 有梯度。**不要**在 `w_t` 处调用 `.detach()`，否则 `sigma_t_raw` 将无法通过光度损失学习。但在梯度闸门中对静态点清零 `sigma_t_raw.grad` 时，**在 `loss.backward()` 之后**执行。

### ⚠️ 注意5：PLY 序列化需保存时域属性
保存模型时，除了原版 3DGS 的属性外，还需保存 `t_mu`、`sigma_t_raw`、`velocity`、`is_static`。建议在 PLY header 中添加自定义属性字段，并在 `load_ply` 中兼容旧版（无时域属性时降级为静态模式）。

### ⚠️ 注意6：大规模数据集的内存管理
对于 150 帧 × 36 视角 = 5400 相机，建议启用延迟图像加载（Lazy Loading）。如果实现 LRU 缓存，缓存大小建议不超过 200 帧（约 2-4 GB GPU 显存）。

### ⚠️ 注意7：FastGS 的 per-splat 并行反传兼容性
FastGS 从 Taming-3DGS 引入了 per-splat 并行反传（替代原版 per-pixel）。梯度闸门中对 `_xyz.grad` 等参数的清零操作，需要确认 per-splat 反传输出的梯度格式与原版一致，否则索引可能错位。建议在测试阶段先用原版反传验证梯度闸门的正确性，再切换到 per-splat。

---

## 验证清单

在实现完成后，按顺序验证以下测试点：

1. **静态硬拉回**：训练 1000 步后，检查 `gaussians._velocity[is_static].abs().max()` 是否为 0。
2. **因果律约束**：以 t=0.1 渲染时，`t_mu > 0.1` 的动态高斯的 `alive_mask` 全为 False。
3. **时域权重梯度**：检查 `sigma_t_raw.grad[~is_static]` 在 backward 后非零。
4. **VCD 时域权重**：对一个 `t_mu=0.8` 的动态高斯，在 t=0.0 的视角下的 VCD 分数贡献应近似为 0。
5. **解耦 opacity reset**：reset 后，`get_opacity()[~is_static]` 的值不应变化。
6. **子代继承**：clone 之后，新创建点的 `is_static` 与父代一致。
7. **CB 顺序**：检查传入光栅化器的高斯数量等于 `alive_mask.sum()`，而非全量 N。
8. **损失下降**：在单帧静态场景（退化为标准 FastGS）上，训练 5K 步的 PSNR 曲线应与原版 FastGS 近似（±0.5 dB 以内）。

---

## 超参数参考

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| K（VCD/VCP 采样视角数） | 10 | 与 FastGS 原版一致 |
| τ_d（静态点致密化阈值） | 5 | 与 FastGS 原版一致 |
| τ_d（动态点致密化阈值） | 2.5 | 动态点视角覆盖稀疏，需降低 |
| τ_p（剪枝阈值） | 0.9 | 与 FastGS 原版一致 |
| τ_alive（因果剪枝阈值） | 0.005 | alpha'(t) 最小有效不透明度 |
| w_t 活跃期阈值（致密化） | 0.2 | 约 ±2σ 生命周期核心内 |
| w_t 当前帧阈值（梯度闸门） | 0.5 | 约 ±1σ 生命周期核心内 |
| σ_t 初始值（动态点） | log(2.5/N_frames) | 初始覆盖约 2.5 帧 |
| σ_t 初始值（静态点） | log(1000) | 全时域可见 |
| λ_v（速度平滑权重） | 0.01 | 可在 0.005-0.05 之间调整 |
| 动态点 opacity reset 值 | 不重置 | 保护动态前景 |
| 静态点 opacity reset 值 | 0.01 | 原版行为 |
| β（CB Mahalanobis 缩放） | 与 FastGS 原版一致 | 动态场景可适当放宽 |

---

## 输出格式

训练完成后，模型保存为：
```
output/
├── point_cloud/
│   ├── iteration_30000/
│   │   └── point_cloud.ply   # 包含时域属性的完整模型
├── renders/
│   ├── frame_0000/
│   │   ├── cam_00.png
│   │   └── ...
│   └── ...
└── cfg_args   # 训练超参数记录
```

PLY 文件中额外属性字段（在原版 3DGS 属性之后追加）：
```
property float t_mu
property float sigma_t_raw
property float vel_x
property float vel_y
property float vel_z
property uchar is_static   # 0 或 1
```

---

*文档版本 v1.0 | 基于 FastGS (arXiv:2511.04283v3) 和 TD-4DGS 内部技术报告*

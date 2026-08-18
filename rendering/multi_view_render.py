"""基于 nvdiffrast 的多视角可微渲染器。

该模块负责把生成器输出的三角网格渲染为多个视角的图像，渲染过程完全可微，
梯度可以从图像损失（如 CLIP 语义损失、2D 判别器损失）反向传播到 mesh 顶点。
"""

from typing import Dict, Optional, Tuple

import math

import torch
import torch.nn.functional as F
from torch import Tensor

try:  # nvdiffrast 依赖 CUDA 编译，导入失败时延迟到实例化再报错
    import nvdiffrast.torch as dr

    _NVDIFFRAST_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    dr = None
    _NVDIFFRAST_AVAILABLE = False


def look_at_matrix(eye: Tensor, target: Tensor, up: Tensor) -> Tensor:
    """构造 world-to-camera 的视图矩阵。

    Args:
        eye: [..., 3] 相机位置（世界坐标）。
        target: [..., 3] 注视点。
        up: [..., 3] 上方向参考向量。

    Returns:
        [..., 4, 4] 视图矩阵（右手系，相机朝 -z 方向看）。
    """
    forward = F.normalize(target - eye, dim=-1)
    right = F.normalize(torch.cross(forward, up, dim=-1), dim=-1)
    true_up = torch.cross(right, forward, dim=-1)

    # 旋转部分为 [right, up, -forward] 的行向量堆叠
    rotation = torch.stack([right, true_up, -forward], dim=-2)  # [..., 3, 3]
    translation = -torch.matmul(rotation, eye.unsqueeze(-1))  # [..., 3, 1]

    view = torch.cat([rotation, translation], dim=-1)  # [..., 3, 4]
    bottom = torch.zeros_like(view[..., :1, :])
    bottom[..., 0, 3] = 1.0
    return torch.cat([view, bottom], dim=-2)  # [..., 4, 4]


def perspective_matrix(
    fov_degrees: float,
    aspect: float = 1.0,
    near: float = 0.1,
    far: float = 100.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """构造 OpenGL 风格的透视投影矩阵 [4, 4]。"""
    tan_half_fov = math.tan(math.radians(fov_degrees) * 0.5)
    proj = torch.zeros(4, 4, device=device, dtype=dtype)
    proj[0, 0] = 1.0 / (aspect * tan_half_fov)
    proj[1, 1] = 1.0 / tan_half_fov
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


def compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """由面法线累加得到顶点法线。

    Args:
        vertices: [B, V, 3] 顶点坐标。
        faces: [F, 3] 面索引（batch 内共享）。

    Returns:
        [B, V, 3] 单位化的顶点法线。
    """
    faces = faces.long()  # scatter_add 要求 int64 索引
    v0 = vertices[:, faces[:, 0]]
    v1 = vertices[:, faces[:, 1]]
    v2 = vertices[:, faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [B, F, 3]

    normals = torch.zeros_like(vertices)
    for i in range(3):
        index = faces[:, i].unsqueeze(0).unsqueeze(-1).expand(
            vertices.shape[0], -1, 3
        )
        normals = normals.scatter_add(1, index, face_normals)
    return F.normalize(normals, dim=-1, eps=1e-8)


class MultiViewRenderer:
    """可微分多视角渲染器，支持梯度反向传播到 mesh 顶点。

    渲染管线：相机采样 -> MVP 变换 -> nvdiffrast 光栅化 -> Phong 着色 -> 抗锯齿。
    输出图像张量的行方向已翻转为常规图像坐标（第 0 行为图像顶部）。
    """

    def __init__(
        self,
        image_size: int = 256,
        num_views: int = 4,
        camera_distance: float = 2.5,
        elevation_range: Tuple[float, float] = (-30, 30),
        azimuth_strategy: str = "stratified",
        fov: float = 40.0,
        near: float = 0.1,
        far: float = 100.0,
        background: float = 0.0,
        ambient: float = 0.3,
        diffuse: float = 0.7,
        specular: float = 0.0,
        shininess: float = 16.0,
        base_color: Tuple[float, float, float] = (0.75, 0.75, 0.75),
        light_direction: Tuple[float, float, float] = (0.0, 0.5, 1.0),
        context_type: str = "auto",
        device: str = "cuda",
    ) -> None:
        """
        Args:
            image_size: 渲染分辨率（正方形）。
            num_views: 每个样本渲染的视角数量。
            camera_distance: 相机到物体中心的距离。
            elevation_range: 仰角采样范围（度）。
            azimuth_strategy: "stratified" | "fixed" | "random"。
            fov: 垂直视场角（度）。
            background: 背景灰度值（0 为黑色背景）。
            ambient/diffuse/specular/shininess: Phong 着色系数。
            base_color: 未提供顶点颜色时使用的默认 albedo。
            light_direction: 世界坐标下的平行光方向（指向物体）。
            context_type: "auto" | "cuda" | "gl"，nvdiffrast 光栅化上下文类型。
            device: 渲染设备，nvdiffrast 仅支持 CUDA。
        """
        if azimuth_strategy not in ("stratified", "fixed", "random"):
            raise ValueError(f"未知的 azimuth_strategy: {azimuth_strategy}")

        self.image_size = image_size
        self.num_views = num_views
        self.camera_distance = camera_distance
        self.elevation_range = elevation_range
        self.azimuth_strategy = azimuth_strategy
        self.fov = fov
        self.near = near
        self.far = far
        self.background = background

        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.base_color = base_color

        self.device = torch.device(device)
        self.light_direction = F.normalize(
            torch.tensor(light_direction, dtype=torch.float32, device=self.device),
            dim=-1,
        )
        self.up_vector = torch.tensor(
            [0.0, 1.0, 0.0], dtype=torch.float32, device=self.device
        )

        self.projection = perspective_matrix(
            fov, 1.0, near, far, device=self.device
        )

        self._context_type = context_type
        self._glctx = None  # 延迟创建，避免非 CUDA 环境下导入即失败

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @property
    def glctx(self):
        """惰性创建并缓存 nvdiffrast 光栅化上下文。"""
        if self._glctx is None:
            if not _NVDIFFRAST_AVAILABLE:
                raise RuntimeError(
                    "未检测到 nvdiffrast，请先安装："
                    "pip install git+https://github.com/NVlabs/nvdiffrast.git"
                )
            if self._context_type == "gl":
                self._glctx = dr.RasterizeGLContext(device=self.device)
            elif self._context_type == "cuda":
                self._glctx = dr.RasterizeCudaContext(device=self.device)
            else:  # auto：优先 GL（支持更大分辨率），失败回退 CUDA
                try:
                    self._glctx = dr.RasterizeGLContext(device=self.device)
                except Exception:  # pragma: no cover - 取决于运行环境
                    self._glctx = dr.RasterizeCudaContext(device=self.device)
        return self._glctx

    def _sample_azimuths(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """按策略采样方位角（度），返回 [B, N_views]。"""
        n = self.num_views
        segment = 360.0 / n
        base = torch.arange(n, device=device, dtype=dtype) * segment  # [N]

        if self.azimuth_strategy == "fixed":
            return base.unsqueeze(0).expand(batch_size, -1).contiguous()

        if self.azimuth_strategy == "stratified":
            # 每段内均匀随机采样一个角度，保证视角覆盖整个 360 度
            jitter = torch.rand(batch_size, n, device=device, dtype=dtype) * segment
            # 随机整体旋转，避免每个 batch 的第一视角总是正面
            offset = torch.rand(batch_size, 1, device=device, dtype=dtype) * segment
            return (base.unsqueeze(0) + jitter + offset) % 360.0

        return torch.rand(batch_size, n, device=device, dtype=dtype) * 360.0

    def _sample_elevations(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """在 elevation_range 内均匀采样仰角（度），返回 [B, N_views]。"""
        low, high = self.elevation_range
        if self.azimuth_strategy == "fixed":
            mid = 0.5 * (low + high)
            return torch.full(
                (batch_size, self.num_views), mid, device=device, dtype=dtype
            )
        rand = torch.rand(batch_size, self.num_views, device=device, dtype=dtype)
        return low + rand * (high - low)

    # ------------------------------------------------------------------ #
    # 相机
    # ------------------------------------------------------------------ #
    def generate_camera_poses(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """生成分层采样的相机位姿。

        azimuth_strategy:
        - "stratified": 将 360 度均匀分为 num_views 段，每段内随机采样
        - "fixed": 固定等间距 (0, 90, 180, 270)
        - "random": 完全随机

        Args:
            batch_size: 批大小。
            device: 输出设备，默认使用渲染器设备。

        Returns:
            camera_poses: [B, N_views, 4, 4] world-to-camera 视图矩阵
            azimuth_angles: [B, N_views] 方位角（弧度）
            elevation_angles: [B, N_views] 仰角（弧度）
        """
        device = self.device if device is None else torch.device(device)
        dtype = torch.float32

        azimuth_deg = self._sample_azimuths(batch_size, device, dtype)
        elevation_deg = self._sample_elevations(batch_size, device, dtype)

        azimuth = torch.deg2rad(azimuth_deg)
        elevation = torch.deg2rad(elevation_deg)

        # 球面坐标 -> 笛卡尔坐标（y 轴向上，azimuth=0 位于 +z 正面）
        cos_el = torch.cos(elevation)
        eye = torch.stack(
            [
                self.camera_distance * cos_el * torch.sin(azimuth),
                self.camera_distance * torch.sin(elevation),
                self.camera_distance * cos_el * torch.cos(azimuth),
            ],
            dim=-1,
        )  # [B, N, 3]

        target = torch.zeros_like(eye)
        up = self.up_vector.to(device).expand_as(eye)
        camera_poses = look_at_matrix(eye, target, up)  # [B, N, 4, 4]
        return camera_poses, azimuth, elevation

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def render(
        self,
        vertices: Tensor,
        faces: Tensor,
        camera_poses: Optional[Tensor] = None,
        vertex_colors: Optional[Tensor] = None,
        return_depth: bool = False,
    ) -> Dict[str, Tensor]:
        """渲染多视角图像。

        Args:
            vertices: [B, V, 3] mesh 顶点。
            faces: [B, F, 3] 或 [F, 3] 面索引。
            camera_poses: 可选 [B, N_views, 4, 4] 视图矩阵，不提供则自动生成。
            vertex_colors: 可选 [B, V, 3] 顶点颜色（albedo），范围 [0, 1]。
            return_depth: 是否额外返回深度图。

        Returns:
            dict with:
            - 'images': [B, N_views, 3, H, W] 渲染图像
            - 'masks': [B, N_views, 1, H, W] 前景 mask
            - 'azimuths': [B, N_views] 方位角（弧度）
            - 'elevations': [B, N_views] 仰角（弧度）
            - 'camera_poses': [B, N_views, 4, 4] 实际使用的视图矩阵
            - 'depths': [B, N_views, 1, H, W]（当 return_depth=True）
        """
        if vertices.dim() != 3 or vertices.shape[-1] != 3:
            raise ValueError(f"vertices 形状应为 [B, V, 3]，实际为 {tuple(vertices.shape)}")

        batch_size = vertices.shape[0]
        device = vertices.device

        if camera_poses is None:
            camera_poses, azimuths, elevations = self.generate_camera_poses(
                batch_size, device=device
            )
        else:
            if camera_poses.dim() != 4:
                raise ValueError(
                    f"camera_poses 形状应为 [B, N_views, 4, 4]，实际为 {tuple(camera_poses.shape)}"
                )
            camera_poses = camera_poses.to(device=device, dtype=vertices.dtype)
            azimuths, elevations = self._recover_angles(camera_poses)

        num_views = camera_poses.shape[1]
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(batch_size, -1, -1)

        images, masks, depths = [], [], []
        # 每个样本的拓扑可能不同（如 DMTet 输出），因此逐样本、批量视角地光栅化
        for b in range(batch_size):
            out = self._render_single(
                vertices[b],
                faces[b],
                camera_poses[b],
                None if vertex_colors is None else vertex_colors[b],
            )
            images.append(out[0])
            masks.append(out[1])
            depths.append(out[2])

        result: Dict[str, Tensor] = {
            "images": torch.stack(images, dim=0),  # [B, N, 3, H, W]
            "masks": torch.stack(masks, dim=0),  # [B, N, 1, H, W]
            "azimuths": azimuths,
            "elevations": elevations,
            "camera_poses": camera_poses,
        }
        if return_depth:
            result["depths"] = torch.stack(depths, dim=0)
        assert result["images"].shape[1] == num_views
        return result

    def render_silhouette(
        self,
        vertices: Tensor,
        faces: Tensor,
        camera_poses: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """仅渲染可微轮廓（silhouette），用于几何监督。

        Returns:
            dict with 'silhouettes': [B, N_views, 1, H, W]，以及视角角度信息。
        """
        out = self.render(vertices, faces, camera_poses)
        return {
            "silhouettes": out["masks"],
            "azimuths": out["azimuths"],
            "elevations": out["elevations"],
            "camera_poses": out["camera_poses"],
        }

    # ------------------------------------------------------------------ #
    # 单样本渲染
    # ------------------------------------------------------------------ #
    def _render_single(
        self,
        vertices: Tensor,
        faces: Tensor,
        camera_poses: Tensor,
        vertex_colors: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """渲染单个 mesh 的所有视角。

        Args:
            vertices: [V, 3]
            faces: [F, 3]
            camera_poses: [N_views, 4, 4]
            vertex_colors: [V, 3] 或 None

        Returns:
            images [N, 3, H, W], masks [N, 1, H, W], depths [N, 1, H, W]
        """
        num_views = camera_poses.shape[0]
        res = self.image_size
        tri = faces.to(torch.int32).contiguous()

        # 顶点齐次坐标 -> 各视角裁剪空间
        ones = torch.ones_like(vertices[:, :1])
        verts_homo = torch.cat([vertices, ones], dim=-1)  # [V, 4]
        mvp = self.projection.to(
            device=vertices.device, dtype=vertices.dtype
        ) @ camera_poses  # [N, 4, 4]
        pos_clip = torch.einsum("nij,vj->nvi", mvp, verts_homo).contiguous()

        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution=[res, res])
        mask = (rast[..., 3:4] > 0).to(vertices.dtype)  # [N, H, W, 1]

        # 着色所需的世界坐标属性
        normals = compute_vertex_normals(vertices.unsqueeze(0), faces)[0]  # [V, 3]
        if vertex_colors is None:
            albedo = torch.tensor(
                self.base_color, dtype=vertices.dtype, device=vertices.device
            ).expand_as(vertices)
        else:
            albedo = vertex_colors.to(vertices.dtype).clamp(0.0, 1.0)

        attrs = torch.cat([vertices, normals, albedo], dim=-1)  # [V, 9]
        attrs = attrs.unsqueeze(0).expand(num_views, -1, -1).contiguous()
        interp, _ = dr.interpolate(attrs, rast, tri)  # [N, H, W, 9]

        world_pos = interp[..., 0:3]
        shade_normal = F.normalize(interp[..., 3:6], dim=-1, eps=1e-8)
        shade_albedo = interp[..., 6:9]

        color = self._phong_shade(world_pos, shade_normal, shade_albedo, camera_poses)
        color = color * mask + self.background * (1.0 - mask)

        # antialias 让轮廓边缘对顶点位置可导，是几何梯度的主要来源
        color = dr.antialias(color, rast, pos_clip, tri)
        mask_aa = dr.antialias(mask, rast, pos_clip, tri)

        images = color.permute(0, 3, 1, 2)  # [N, 3, H, W]
        masks = mask_aa.permute(0, 3, 1, 2)  # [N, 1, H, W]
        depths = rast[..., 2:3].permute(0, 3, 1, 2) * masks

        # nvdiffrast 输出行序为自下而上，翻转为常规图像坐标
        images = torch.flip(images, dims=[-2])
        masks = torch.flip(masks, dims=[-2])
        depths = torch.flip(depths, dims=[-2])
        return images.clamp(0.0, 1.0), masks.clamp(0.0, 1.0), depths

    def _phong_shade(
        self,
        world_pos: Tensor,
        normals: Tensor,
        albedo: Tensor,
        camera_poses: Tensor,
    ) -> Tensor:
        """对插值后的属性做 Phong 着色，返回 [N, H, W, 3]。"""
        light_dir = self.light_direction.to(
            device=normals.device, dtype=normals.dtype
        ).view(1, 1, 1, 3)

        lambert = torch.clamp((normals * light_dir).sum(-1, keepdim=True), min=0.0)
        color = albedo * (self.ambient + self.diffuse * lambert)

        if self.specular > 0.0:
            # 相机位置由视图矩阵反解：C = -R^T t
            rotation = camera_poses[:, :3, :3]
            translation = camera_poses[:, :3, 3:]
            cam_pos = -torch.matmul(rotation.transpose(-1, -2), translation)
            cam_pos = cam_pos.squeeze(-1).view(-1, 1, 1, 3)

            view_dir = F.normalize(cam_pos - world_pos, dim=-1, eps=1e-8)
            half_dir = F.normalize(view_dir + light_dir, dim=-1, eps=1e-8)
            spec = torch.clamp((normals * half_dir).sum(-1, keepdim=True), min=0.0)
            color = color + self.specular * torch.pow(spec, self.shininess)

        return color

    # ------------------------------------------------------------------ #
    # 角度反解
    # ------------------------------------------------------------------ #
    def _recover_angles(self, camera_poses: Tensor) -> Tuple[Tensor, Tensor]:
        """从外部传入的视图矩阵反解方位角与仰角（弧度）。

        Args:
            camera_poses: [B, N_views, 4, 4] world-to-camera 矩阵。

        Returns:
            azimuths [B, N_views], elevations [B, N_views]
        """
        rotation = camera_poses[..., :3, :3]
        translation = camera_poses[..., :3, 3:]
        cam_pos = -torch.matmul(rotation.transpose(-1, -2), translation).squeeze(-1)

        x, y, z = cam_pos[..., 0], cam_pos[..., 1], cam_pos[..., 2]
        radius = torch.linalg.norm(cam_pos, dim=-1).clamp(min=1e-8)
        azimuth = torch.atan2(x, z) % (2.0 * math.pi)
        elevation = torch.asin((y / radius).clamp(-1.0, 1.0))
        return azimuth, elevation

"""异常3D结构负样本程序化构造模块。

用于训练 SemanticCritic 区分正常/异常结构：给定一个「正常」的三角网格，
程序化地施加一种语义层面不合理的变形（多头、对称性破坏、不自然突起、多腿），
得到可直接送入渲染器 / 判别器的负样本。

约定与项目其余部分保持一致：
- ``vertices``: [V, 3] float32
- ``faces``: [F, 3] int64（三角面片，零基索引）

可微性说明：
- ``mirror_artifact`` / ``random_extrusion`` 只改动顶点位置，梯度可回传到输入顶点；
- ``local_duplication`` / ``topology_duplication`` 会增删顶点与面片，拓扑操作本身
  不可微（新增顶点仍由原顶点通过可微运算得到，故对原顶点的梯度路径依然存在）。
  异常样本只作为 Critic 的判别目标使用，无需反传到构造过程。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["STRATEGIES", "compute_vertex_normals", "AnomalyConstructor"]

EPS = 1e-8

#: 支持的异常构造策略
STRATEGIES: Tuple[str, ...] = (
    "local_duplication",
    "mirror_artifact",
    "random_extrusion",
    "topology_duplication",
)

#: 会改变拓扑（顶点/面片数量）的策略
_TOPOLOGY_STRATEGIES: Tuple[str, ...] = (
    "local_duplication",
    "topology_duplication",
)


# ---------------------------------------------------------------------- #
# 基础几何工具
# ---------------------------------------------------------------------- #
def _canonicalize(vertices: Tensor, faces: Tensor) -> Tuple[Tensor, Tensor]:
    """把输入统一成 vertices=[V, 3] float、faces=[F, 3] long，并剔除非法面片。

    允许传入 [1, V, 3] / [1, F, 3] 的单样本 batch 形式（与 generator 输出兼容）。
    """
    if vertices.dim() == 3:
        if vertices.size(0) != 1:
            raise ValueError(
                f"construct 只接受单个 mesh，batch 请用 construct_batch，"
                f"得到 {tuple(vertices.shape)}"
            )
        vertices = vertices[0]
    if vertices.dim() != 2 or vertices.size(-1) != 3:
        raise ValueError(f"vertices 应为 [V, 3]，得到 {tuple(vertices.shape)}")

    if faces.dim() == 3:
        faces = faces[0]
    if faces.numel() == 0:
        faces = faces.reshape(0, 3)
    if faces.dim() != 2 or faces.size(-1) != 3:
        raise ValueError(f"faces 应为 [F, 3]，得到 {tuple(faces.shape)}")

    verts = vertices.float()
    faces = faces.long()
    if faces.numel() > 0:
        num_verts = verts.shape[0]
        # 越界索引会让后续 index_select 直接崩掉，这里静默丢弃
        valid = (faces >= 0).all(dim=1) & (faces < num_verts).all(dim=1)
        # 退化面（重复顶点）无法贡献法线，一并丢弃
        non_degenerate = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        keep = valid & non_degenerate
        if not bool(keep.all()):
            faces = faces[keep]
    return verts, faces


def compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """计算单位顶点法线（面积加权的面法线累加）。

    Args:
        vertices: [V, 3]
        faces: [F, 3]

    Returns:
        [V, 3] 单位法线。无面片（纯点云）或孤立顶点退化为「由质心指向该点」的
        径向方向，保证返回值始终是有意义的单位向量。
    """
    centroid = vertices.mean(dim=0, keepdim=True) if vertices.numel() > 0 else vertices
    radial = F.normalize(vertices - centroid, dim=-1, eps=EPS)
    if faces.numel() == 0 or vertices.numel() == 0:
        return radial

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    # 不归一化面法线 => 天然按三角形面积加权
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    normals = torch.zeros_like(vertices)
    normals = normals.index_add(
        0,
        faces.reshape(-1),
        face_normals.repeat_interleave(3, dim=0),
    )
    lengths = normals.norm(dim=-1, keepdim=True)
    normals = F.normalize(normals, dim=-1, eps=EPS)
    # 未被任何面片覆盖的顶点回退到径向方向
    return torch.where(lengths > EPS, normals, radial)


def _bbox_diagonal(vertices: Tensor) -> float:
    """mesh 包围盒对角线长度，作为位移幅度的尺度基准。"""
    if vertices.numel() == 0:
        return 1.0
    extent = vertices.max(dim=0).values - vertices.min(dim=0).values
    diag = float(extent.norm().item())
    return diag if diag > EPS else 1.0


# ---------------------------------------------------------------------- #
# 主体
# ---------------------------------------------------------------------- #
class AnomalyConstructor:
    """程序化构造3D mesh异常负样本。

    支持4种异常策略：
    - local_duplication: 复制局部区域模拟"多头/多脸"
    - mirror_artifact: 局部镜像翻转模拟对称性破坏
    - random_extrusion: 法线方向随机凸起模拟不自然突起
    - topology_duplication: 复制连通面片并平移模拟"多腿"

    Args:
        strategies: 启用的策略列表，None 表示全部启用
        severity_range: (min, max) 异常严重程度范围
        region_ratio: (min, max) 单次异常涉及的顶点比例范围
        num_leg_copies: ``topology_duplication`` 复制底部区域的份数
        up_axis: 竖直轴索引（0/1/2），决定"底部区域"的方向，默认 1（y 轴向上）
        seed: 随机种子，None 表示不固定
    """

    def __init__(
        self,
        strategies: Optional[Sequence[str]] = None,
        severity_range: Tuple[float, float] = (0.3, 0.8),
        region_ratio: Tuple[float, float] = (0.08, 0.25),
        num_leg_copies: int = 2,
        up_axis: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        enabled = tuple(STRATEGIES if strategies is None else strategies)
        unknown = [s for s in enabled if s not in STRATEGIES]
        if unknown:
            raise ValueError(f"未知的异常策略: {unknown}，可选: {list(STRATEGIES)}")
        if not enabled:
            raise ValueError("strategies 不能为空")

        low, high = float(severity_range[0]), float(severity_range[1])
        if not 0.0 <= low <= high:
            raise ValueError(f"severity_range 非法: {severity_range}")
        r_low, r_high = float(region_ratio[0]), float(region_ratio[1])
        if not 0.0 < r_low <= r_high <= 1.0:
            raise ValueError(f"region_ratio 非法: {region_ratio}")
        if up_axis not in (0, 1, 2):
            raise ValueError(f"up_axis 应为 0/1/2，得到 {up_axis}")

        self.strategies: Tuple[str, ...] = enabled
        self.severity_range: Tuple[float, float] = (low, high)
        self.region_ratio: Tuple[float, float] = (r_low, r_high)
        self.num_leg_copies = max(1, int(num_leg_copies))
        self.up_axis = up_axis
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def construct(
        self,
        vertices: Tensor,
        faces: Tensor,
        strategy: Optional[str] = None,
        severity: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """对输入mesh施加一种异常变换。

        Args:
            vertices: (V, 3) FloatTensor
            faces: (F, 3) LongTensor
            strategy: str or None (随机选择)
            severity: float or None (从severity_range中随机)

        Returns:
            anomaly_vertices: (V', 3) FloatTensor (顶点数可能变化)
            anomaly_faces: (F', 3) LongTensor
            anomaly_info: dict with keys 'strategy'、'severity'、
                'affected_vertices'，以及附加统计 'requested_strategy'、
                'added_vertices'、'added_faces'、'num_vertices'、'num_faces'
        """
        verts, tri = _canonicalize(vertices, faces)
        requested = strategy
        if strategy is None:
            strategy = str(self._rng.choice(np.asarray(self.strategies)))
        elif strategy not in STRATEGIES:
            raise ValueError(f"未知的异常策略: {strategy!r}")

        if severity is None:
            severity = float(self._rng.uniform(*self.severity_range))
        severity = float(np.clip(severity, 0.0, 1.0))

        strategy = self._resolve_strategy(strategy, verts, tri)
        if strategy == "identity":
            # 空 mesh / 顶点过少：无法构造任何有意义的异常，原样返回
            return (
                verts.clone(),
                tri.clone(),
                self._make_info("identity", requested, 0.0, 0, verts, tri, verts, tri),
            )

        handler = {
            "local_duplication": self._local_duplication,
            "mirror_artifact": self._mirror_artifact,
            "random_extrusion": self._random_extrusion,
            "topology_duplication": self._topology_duplication,
        }[strategy]
        new_verts, new_faces, affected = handler(verts, tri, severity)

        if affected == 0 and strategy in _TOPOLOGY_STRATEGIES:
            # 局部区域凑不出完整面片（如面片过于稀疏）时退化为纯位移异常
            strategy = "random_extrusion"
            new_verts, new_faces, affected = self._random_extrusion(
                verts, tri, severity
            )

        info = self._make_info(
            strategy, requested, severity, affected, verts, tri, new_verts, new_faces
        )
        return new_verts, new_faces, info

    def construct_batch(
        self,
        vertices_list: Sequence[Tensor],
        faces_list: Sequence[Tensor],
        num_anomalies_per_sample: int = 1,
    ) -> Tuple[List[Tensor], List[Tensor], List[Dict[str, Any]]]:
        """批量构造异常样本。

        Args:
            vertices_list: List of (V_i, 3) tensors
            faces_list: List of (F_i, 3) tensors（长度为 1 时视为 batch 共享拓扑）
            num_anomalies_per_sample: 每个样本构造几个异常版本

        Returns:
            anomaly_vertices_list: List of tensors
            anomaly_faces_list: List of tensors
            anomaly_infos: List of dicts（顺序为 sample0 的全部异常、sample1 ...，
                每个 info 额外含 'sample_index'）
        """
        if num_anomalies_per_sample < 1:
            raise ValueError(
                f"num_anomalies_per_sample 至少为 1，得到 {num_anomalies_per_sample}"
            )
        shared_faces = len(faces_list) == 1 and len(vertices_list) > 1
        if not shared_faces and len(vertices_list) != len(faces_list):
            raise ValueError(
                f"vertices_list ({len(vertices_list)}) 与 faces_list "
                f"({len(faces_list)}) 长度不一致"
            )

        out_verts: List[Tensor] = []
        out_faces: List[Tensor] = []
        infos: List[Dict[str, Any]] = []
        for i, verts in enumerate(vertices_list):
            tri = faces_list[0] if shared_faces else faces_list[i]
            for _ in range(num_anomalies_per_sample):
                a_verts, a_faces, info = self.construct(verts, tri)
                info["sample_index"] = i
                out_verts.append(a_verts)
                out_faces.append(a_faces)
                infos.append(info)
        return out_verts, out_faces, infos

    # ------------------------------------------------------------------ #
    # 策略实现
    # ------------------------------------------------------------------ #
    def _local_duplication(
        self, verts: Tensor, faces: Tensor, severity: float
    ) -> Tuple[Tensor, Tensor, int]:
        """复制局部区域并沿法线推开，模拟"多头/多脸"。"""
        num_target = self._region_size(verts.shape[0])
        seed = self._pick_seed(faces, verts.shape[0])
        region = self._region_by_bfs(verts, faces, seed, num_target)

        sub_verts, sub_faces, used = self._extract_submesh(verts, faces, region)
        if sub_faces.numel() == 0:
            return verts.clone(), faces.clone(), 0

        normals = compute_vertex_normals(verts, faces)
        direction = F.normalize(normals[used].mean(dim=0), dim=0, eps=EPS)
        if float(direction.norm().item()) < 1e-3:
            # 区域法线相互抵消（如环状区域）时改用"远离质心"的方向
            direction = F.normalize(
                sub_verts.mean(dim=0) - verts.mean(dim=0), dim=0, eps=EPS
            )
        if float(direction.norm().item()) < 1e-3:
            direction = self._random_unit(verts)

        diag = _bbox_diagonal(verts)
        offset = direction * (0.25 + 0.75 * severity) * diag * 0.5
        # 复制体略微缩放，避免与原区域完全重合导致"看起来只是平移"
        scale = float(self._rng.uniform(0.7, 1.15))
        sub_center = sub_verts.mean(dim=0, keepdim=True)
        dup_verts = sub_center + (sub_verts - sub_center) * scale + offset

        new_verts = torch.cat([verts, dup_verts], dim=0)
        new_faces = torch.cat([faces, sub_faces + verts.shape[0]], dim=0)
        return new_verts, new_faces, int(used.numel())

    def _mirror_artifact(
        self, verts: Tensor, faces: Tensor, severity: float
    ) -> Tuple[Tensor, Tensor, int]:
        """把一侧顶点子集向镜像位置混合，破坏原有对称性（不改变拓扑）。"""
        # 竖直轴上做镜像会把物体压扁，只在水平轴上做
        axes = [a for a in (0, 1, 2) if a != self.up_axis]
        axis = int(self._rng.choice(axes))
        center = verts.mean(dim=0)
        sign = 1.0 if self._rng.rand() < 0.5 else -1.0

        side = ((verts[:, axis] - center[axis]) * sign) > 0
        side_idx = side.nonzero(as_tuple=False).squeeze(1)
        if side_idx.numel() == 0:
            side_idx = torch.arange(verts.shape[0], device=verts.device)

        # 只取该侧的一个子集 => 局部而非整体镜像，异常感更强
        keep_ratio = float(self._rng.uniform(0.4, 1.0))
        num_keep = max(1, int(round(side_idx.numel() * keep_ratio)))
        if num_keep < side_idx.numel():
            choice = self._rng.choice(side_idx.numel(), num_keep, replace=False)
            side_idx = side_idx[torch.as_tensor(choice, device=verts.device).long()]

        delta = torch.zeros_like(verts)
        # v_new = (1-s)*v + s*v_mirror，镜像面过质心 => v_mirror_a = 2*c_a - v_a
        delta[side_idx, axis] = 2.0 * severity * (center[axis] - verts[side_idx, axis])
        return verts + delta, faces.clone(), int(side_idx.numel())

    def _random_extrusion(
        self, verts: Tensor, faces: Tensor, severity: float
    ) -> Tuple[Tensor, Tensor, int]:
        """随机顶点沿法线方向凸起，模拟不自然突起（不改变拓扑）。"""
        num_verts = verts.shape[0]
        num_pick = self._region_size(num_verts)
        choice = self._rng.choice(num_verts, num_pick, replace=False)
        idx = torch.as_tensor(choice, device=verts.device).long()

        normals = compute_vertex_normals(verts, faces)
        diag = _bbox_diagonal(verts)
        # 每个顶点幅度带随机抖动，形成参差不齐的尖刺
        jitter = torch.as_tensor(
            self._rng.uniform(0.5, 1.5, size=num_pick),
            device=verts.device,
            dtype=verts.dtype,
        ).unsqueeze(1)
        magnitude = severity * diag * 0.3 * jitter

        delta = torch.zeros_like(verts)
        delta[idx] = normals[idx] * magnitude
        return verts + delta, faces.clone(), num_pick

    def _topology_duplication(
        self, verts: Tensor, faces: Tensor, severity: float
    ) -> Tuple[Tensor, Tensor, int]:
        """复制底部区域面片并水平平移，模拟"多腿"效果。"""
        axis = self.up_axis
        num_verts = verts.shape[0]
        num_target = self._region_size(num_verts)

        # 取竖直坐标最低的 k% 顶点作为"腿部"区域
        order = torch.argsort(verts[:, axis])
        region = order[:num_target]
        sub_verts, sub_faces, used = self._extract_submesh(verts, faces, region)
        if sub_faces.numel() == 0:
            return verts.clone(), faces.clone(), 0

        horizontal = [a for a in (0, 1, 2) if a != axis]
        extent = verts.max(dim=0).values - verts.min(dim=0).values
        span = float(max(extent[horizontal[0]].item(), extent[horizontal[1]].item()))
        span = span if span > EPS else _bbox_diagonal(verts)
        distance = (0.3 + 0.7 * severity) * span * 0.5

        base_angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        dup_verts: List[Tensor] = []
        dup_faces: List[Tensor] = []
        cursor = num_verts
        for copy_idx in range(self.num_leg_copies):
            angle = base_angle + 2.0 * np.pi * copy_idx / self.num_leg_copies
            shift = torch.zeros(3, device=verts.device, dtype=verts.dtype)
            shift[horizontal[0]] = distance * float(np.cos(angle))
            shift[horizontal[1]] = distance * float(np.sin(angle))
            dup_verts.append(sub_verts + shift)
            dup_faces.append(sub_faces + cursor)
            cursor += sub_verts.shape[0]

        new_verts = torch.cat([verts] + dup_verts, dim=0)
        new_faces = torch.cat([faces] + dup_faces, dim=0)
        return new_verts, new_faces, int(used.numel())

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _resolve_strategy(self, strategy: str, verts: Tensor, faces: Tensor) -> str:
        """根据 mesh 的退化程度把策略降级到可执行的那一个。"""
        num_verts = verts.shape[0]
        if num_verts == 0:
            return "identity"
        if faces.shape[0] == 0 or num_verts < 4:
            # 没有可用面片时无法抽取子网格，只能做纯顶点位移
            if strategy in _TOPOLOGY_STRATEGIES:
                return "random_extrusion"
        return strategy

    def _region_size(self, num_verts: int) -> int:
        """按 region_ratio 采样局部区域的顶点数量（至少 3 个以凑出面片）。"""
        ratio = float(self._rng.uniform(*self.region_ratio))
        num = int(round(num_verts * ratio))
        return int(np.clip(num, min(3, num_verts), num_verts))

    def _pick_seed(self, faces: Tensor, num_verts: int) -> int:
        """随机挑一个种子顶点，优先落在有面片覆盖的顶点上。"""
        if faces.numel() > 0:
            flat = faces.reshape(-1)
            return int(flat[int(self._rng.randint(flat.numel()))].item())
        return int(self._rng.randint(num_verts))

    def _random_unit(self, verts: Tensor) -> Tensor:
        """随机单位方向向量（与 verts 同 device/dtype）。"""
        vec = torch.as_tensor(
            self._rng.randn(3), device=verts.device, dtype=verts.dtype
        )
        return F.normalize(vec, dim=0, eps=EPS)

    @staticmethod
    def _bidirectional_edges(faces: Tensor) -> Tuple[Tensor, Tensor]:
        """由面片抽取双向边 (src, dst)，用于 BFS 邻域扩张。"""
        edges = torch.cat(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
        )
        return (
            torch.cat([edges[:, 0], edges[:, 1]], dim=0),
            torch.cat([edges[:, 1], edges[:, 0]], dim=0),
        )

    def _region_by_bfs(
        self, verts: Tensor, faces: Tensor, seed: int, num_target: int
    ) -> Tensor:
        """从种子顶点沿 mesh 邻接关系 BFS 扩张，得到一个连通的局部区域。

        当区域所在连通分量不足 ``num_target`` 个顶点时，用欧氏距离最近的
        未访问顶点补齐（对应 mesh 本身破碎的情况）。

        Returns:
            [k] LongTensor 顶点索引
        """
        num_verts = verts.shape[0]
        num_target = int(np.clip(num_target, 1, num_verts))
        visited = torch.zeros(num_verts, dtype=torch.bool, device=verts.device)
        visited[seed] = True
        count = 1

        if faces.numel() > 0:
            src, dst = self._bidirectional_edges(faces)
            while count < num_target:
                candidates = dst[visited[src]]
                candidates = candidates[~visited[candidates]]
                if candidates.numel() == 0:
                    break
                candidates = torch.unique(candidates)
                if count + int(candidates.numel()) > num_target:
                    candidates = candidates[: num_target - count]
                visited[candidates] = True
                count += int(candidates.numel())

        if count < num_target:
            distance = (verts - verts[seed]).norm(dim=-1)
            distance = distance.masked_fill(visited, float("inf"))
            extra = torch.topk(distance, num_target - count, largest=False).indices
            visited[extra] = True
        return visited.nonzero(as_tuple=False).squeeze(1)

    @staticmethod
    def _extract_submesh(
        verts: Tensor, faces: Tensor, region: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """抽取 region 内部的子网格（三个顶点都在区域内的面片）。

        Returns:
            (sub_verts [k, 3], sub_faces [m, 3] 重映射后的索引, used [k] 原索引)
            当区域内没有完整面片时返回三个空张量。
        """
        num_verts = verts.shape[0]
        empty_v = verts.new_zeros((0, 3))
        empty_f = faces.new_zeros((0, 3))
        if faces.numel() == 0 or region.numel() == 0:
            return empty_v, empty_f, faces.new_zeros((0,))

        mask = torch.zeros(num_verts, dtype=torch.bool, device=verts.device)
        mask[region] = True
        sub_faces = faces[mask[faces].all(dim=1)]
        if sub_faces.numel() == 0:
            return empty_v, empty_f, faces.new_zeros((0,))

        used = torch.unique(sub_faces)
        remap = torch.full((num_verts,), -1, dtype=torch.long, device=verts.device)
        remap[used] = torch.arange(used.numel(), device=verts.device)
        return verts[used], remap[sub_faces], used

    @staticmethod
    def _make_info(
        strategy: str,
        requested: Optional[str],
        severity: float,
        affected: int,
        verts: Tensor,
        faces: Tensor,
        new_verts: Tensor,
        new_faces: Tensor,
    ) -> Dict[str, Any]:
        """组装异常样本的元信息。"""
        return {
            "strategy": strategy,
            "severity": severity,
            "affected_vertices": int(affected),
            "requested_strategy": requested,
            "added_vertices": int(new_verts.shape[0] - verts.shape[0]),
            "added_faces": int(new_faces.shape[0] - faces.shape[0]),
            "num_vertices": int(new_verts.shape[0]),
            "num_faces": int(new_faces.shape[0]),
            "changed_topology": strategy in _TOPOLOGY_STRATEGIES,
        }


# ---------------------------------------------------------------------- #
# 自测入口
# ---------------------------------------------------------------------- #
def _demo_sphere(num_rings: int = 16, num_sectors: int = 24) -> Tuple[Tensor, Tensor]:
    """构造一个 UV 球用于自测（纯本地实现，不依赖外部 generator）。"""
    theta = torch.linspace(0.0, np.pi, num_rings + 2)[1:-1]
    phi = torch.arange(num_sectors, dtype=torch.float32) * (2.0 * np.pi / num_sectors)
    x = torch.sin(theta)[:, None] * torch.cos(phi)[None, :]
    y = torch.cos(theta)[:, None].expand(num_rings, num_sectors)
    z = torch.sin(theta)[:, None] * torch.sin(phi)[None, :]
    verts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    faces: List[List[int]] = []
    for r in range(num_rings - 1):
        for s in range(num_sectors):
            v00 = r * num_sectors + s
            v01 = r * num_sectors + (s + 1) % num_sectors
            v10 = (r + 1) * num_sectors + s
            v11 = (r + 1) * num_sectors + (s + 1) % num_sectors
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])
    return verts, torch.tensor(faces, dtype=torch.long)


def _main() -> None:
    """打印各策略的构造统计信息。"""
    torch.manual_seed(0)
    verts, faces = _demo_sphere()
    print("=" * 78)
    print(f"输入 mesh: V={verts.shape[0]}, F={faces.shape[0]}, "
          f"bbox 对角线={_bbox_diagonal(verts):.4f}")
    print("=" * 78)

    constructor = AnomalyConstructor(seed=42)

    header = (
        f"{'strategy':<22}{'sev':>6}{'V_out':>7}{'F_out':>8}"
        f"{'+V':>7}{'+F':>7}{'affected':>10}{'d_bbox':>9}{'可微':>7}"
    )
    print(header)
    print("-" * len(header))
    base_diag = _bbox_diagonal(verts)
    for strategy in STRATEGIES:
        for severity in (0.3, 0.6, 0.9):
            src = verts.clone().requires_grad_(True)
            a_verts, a_faces, info = constructor.construct(
                src, faces, strategy=strategy, severity=severity
            )
            differentiable = a_verts.requires_grad and a_verts.grad_fn is not None
            print(
                f"{info['strategy']:<22}{info['severity']:>6.2f}"
                f"{info['num_vertices']:>7}{info['num_faces']:>8}"
                f"{info['added_vertices']:>7}{info['added_faces']:>7}"
                f"{info['affected_vertices']:>10}"
                f"{_bbox_diagonal(a_verts) - base_diag:>9.3f}"
                f"{'是' if differentiable else '否':>7}"
            )

    # 梯度确实能回传到输入顶点
    src = verts.clone().requires_grad_(True)
    a_verts, _, _ = constructor.construct(src, faces, strategy="random_extrusion")
    a_verts.pow(2).sum().backward()
    grad_norm = float(src.grad.norm().item())
    print(f"\n梯度检查: random_extrusion 输入顶点梯度范数 = {grad_norm:.4f}")

    # 批量构造：随机策略 + 变长 mesh
    small_v, small_f = _demo_sphere(num_rings=6, num_sectors=8)
    v_list, f_list, infos = constructor.construct_batch(
        [verts, small_v], [faces, small_f], num_anomalies_per_sample=3
    )
    print(f"\n批量构造: 2 个样本 x 3 个异常 -> {len(v_list)} 个负样本")
    for a_v, a_f, info in zip(v_list, f_list, infos):
        print(
            f"  sample {info['sample_index']}  {info['strategy']:<22}"
            f"sev={info['severity']:.2f}  V={a_v.shape[0]:<6}F={a_f.shape[0]:<6}"
            f"affected={info['affected_vertices']}"
        )
    counts: Dict[str, int] = {}
    for info in infos:
        counts[info["strategy"]] = counts.get(info["strategy"], 0) + 1
    print(f"  策略分布: {counts}")

    # 边界情况
    print("\n边界情况:")
    cases: List[Tuple[str, Tensor, Tensor]] = [
        ("空 mesh", torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long)),
        ("无面片点云 (3 顶点)", torch.randn(3, 3), torch.zeros(0, 3, dtype=torch.long)),
        (
            "单个三角形",
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0, 1, 2]], dtype=torch.long),
        ),
        (
            "越界/退化面片",
            torch.randn(8, 3),
            torch.tensor([[0, 1, 2], [0, 0, 1], [3, 99, 4], [4, 5, 6]], dtype=torch.long),
        ),
        ("batch 形式 [1, V, 3]", verts.unsqueeze(0), faces.unsqueeze(0)),
    ]
    for name, case_v, case_f in cases:
        for strategy in STRATEGIES:
            a_v, a_f, info = constructor.construct(case_v, case_f, strategy=strategy)
            assert a_f.numel() == 0 or int(a_f.max().item()) < a_v.shape[0], "面索引越界"
            print(
                f"  {name:<22}{strategy:<22}-> {info['strategy']:<22}"
                f"V={a_v.shape[0]:<6}F={a_f.shape[0]:<6}"
                f"affected={info['affected_vertices']}"
            )
    print("\n全部检查通过。")


if __name__ == "__main__":
    _main()

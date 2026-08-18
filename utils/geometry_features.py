"""Mesh 几何统计特征提取（Semantic Critic 的几何输入分支）。

Critic 需要一路与 CLIP 语义互补的「结构」信号。这里采用与判别器解耦的
方案：直接从 mesh 顶点/面片计算一组固定维度的几何描述子，不依赖
``UnifiedDiscriminator`` 的中间层，因此判别器接口无需改动。

描述子由两部分拼成（总维度 == ``geo_dim``）：

1. ``NUM_SCALAR_FEATURES`` 个全局标量统计（顶点/面片规模、包围盒比例、
   径向距离、边长、面积、曲率、质心偏移等），多数经过尺度归一化或
   ``tanh`` 压缩，保证不同拓扑的 mesh 落在可比的数值范围内；
2. 三组**软直方图**（边长分布、曲率分布、径向距离分布），把「局部尖刺」
   「多出来的一团结构」这类异常反映在分布形状上——单看均值/方差往往被
   大量正常顶点稀释，直方图则能保留尾部。

可微性：全部运算（含软直方图的高斯核分配）对顶点坐标可导，因此
Generator 步中 Critic 的梯度能经几何分支回传到顶点；``faces`` 只作为索引
使用。硬分箱不可微，故直方图采用高斯核软分配。

约定与项目其余部分一致：``vertices`` [V, 3] float，``faces`` [F, 3] int64。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    "NUM_SCALAR_FEATURES",
    "MIN_GEO_DIM",
    "MAX_HISTOGRAM_SAMPLES",
    "mesh_geometry_features",
    "batch_geometry_features",
]

EPS = 1e-8

#: 全局标量统计的数量
NUM_SCALAR_FEATURES = 17

#: 至少要为三组直方图各留 1 个 bin
MIN_GEO_DIM = NUM_SCALAR_FEATURES + 3

#: 单组软直方图参与统计的最大样本数（限制 [N, num_bins] 中间张量的显存占用）
MAX_HISTOGRAM_SAMPLES = 4096


# ---------------------------------------------------------------------- #
# 基础工具
# ---------------------------------------------------------------------- #
def _histogram_bins(geo_dim: int) -> Tuple[int, int, int]:
    """把 ``geo_dim - NUM_SCALAR_FEATURES`` 个维度分给三组直方图。

    余数依次分配给前面的直方图，保证拼接结果精确等于 ``geo_dim``（不做零填充）。
    """
    if geo_dim < MIN_GEO_DIM:
        raise ValueError(
            f"geo_dim 至少为 {MIN_GEO_DIM}（{NUM_SCALAR_FEATURES} 个标量 + 3 个直方图 bin），"
            f"得到 {geo_dim}"
        )
    budget = geo_dim - NUM_SCALAR_FEATURES
    base, remainder = divmod(budget, 3)
    return tuple(base + (1 if i < remainder else 0) for i in range(3))  # type: ignore[return-value]


def _soft_histogram(
    values: Tensor,
    low: float,
    high: float,
    num_bins: int,
    sigma_scale: float = 0.75,
    max_samples: int = MAX_HISTOGRAM_SAMPLES,
) -> Tensor:
    """高斯核软直方图（可微），返回 [num_bins] 且各 bin 之和为 1。

    Args:
        values: [N] 待统计的标量序列。
        low/high: 分箱范围，超出范围的值先 clamp 到边界。
        num_bins: bin 数量。
        sigma_scale: 高斯核带宽相对 bin 间距的比例。
        max_samples: 参与统计的最大样本数。软分配的中间张量是 [N, num_bins]，
            高分辨率 mesh 下会成为显存热点，超出上限时按固定步长抽稀
            （faces 顺序在 mesh 上大致均匀，抽稀几乎不改变分布形状）。
    """
    if num_bins <= 0:
        return values.new_zeros(0)
    if values.numel() == 0:
        return values.new_zeros(num_bins)
    if values.numel() > max_samples:
        values = values[:: values.numel() // max_samples + 1]

    centers = torch.linspace(low, high, num_bins, device=values.device, dtype=values.dtype)
    width = (high - low) / max(num_bins - 1, 1)
    sigma = max(width * sigma_scale, EPS)

    clamped = values.clamp(low, high).reshape(-1, 1)
    weights = torch.exp(-0.5 * ((clamped - centers.reshape(1, -1)) / sigma) ** 2)
    # 逐样本归一化 => 每个值向各 bin 贡献总量恒为 1，直方图与 N 无关
    weights = weights / (weights.sum(dim=1, keepdim=True) + EPS)
    return weights.mean(dim=0)


def _uniform_laplacian_magnitude(vertices: Tensor, faces: Tensor) -> Tensor:
    """均匀 Laplacian 模长，作为逐顶点曲率代理，返回 [V]。

    ``||v - mean(neighbors)||`` 在平坦区域接近 0，在尖刺 / 折痕处显著增大，
    比二面角便宜且完全可微。孤立顶点（不被任何面片覆盖）取 0。
    """
    num_vertices = vertices.shape[0]
    if num_vertices == 0 or faces.numel() == 0:
        return vertices.new_zeros(num_vertices)

    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    src = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
    dst = torch.cat([edges[:, 1], edges[:, 0]], dim=0)

    neighbor_sum = torch.zeros_like(vertices).index_add(
        0, src, vertices.index_select(0, dst)
    )
    counts = torch.zeros(
        num_vertices, device=vertices.device, dtype=vertices.dtype
    ).index_add(0, src, torch.ones_like(src, dtype=vertices.dtype))

    mean_neighbor = neighbor_sum / counts.clamp(min=1.0).unsqueeze(-1)
    delta = (vertices - mean_neighbor) * (counts > 0).unsqueeze(-1)
    return delta.norm(dim=-1)


def _edge_lengths(vertices: Tensor, faces: Tensor) -> Tensor:
    """所有面内边的长度 [3F]（内部边被两个面各计一次，不影响分布形状）。"""
    if faces.numel() == 0:
        return vertices.new_zeros(0)
    v0 = vertices.index_select(0, faces[:, 0])
    v1 = vertices.index_select(0, faces[:, 1])
    v2 = vertices.index_select(0, faces[:, 2])
    return torch.cat(
        [(v1 - v0).norm(dim=-1), (v2 - v1).norm(dim=-1), (v0 - v2).norm(dim=-1)], dim=0
    )


def _face_areas(vertices: Tensor, faces: Tensor) -> Tensor:
    """逐面片面积 [F]。"""
    if faces.numel() == 0:
        return vertices.new_zeros(0)
    v0 = vertices.index_select(0, faces[:, 0])
    v1 = vertices.index_select(0, faces[:, 1])
    v2 = vertices.index_select(0, faces[:, 2])
    return 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)


def _mean(values: Tensor) -> Tensor:
    """空张量安全的均值。"""
    if values.numel() == 0:
        return values.new_zeros(())
    return values.mean()


def _std(values: Tensor) -> Tensor:
    """空张量 / 单元素安全的标准差（有偏估计，避免 NaN）。"""
    if values.numel() < 2:
        return values.new_zeros(())
    return values.std(unbiased=False)


def _max(values: Tensor) -> Tensor:
    """空张量安全的最大值。"""
    if values.numel() == 0:
        return values.new_zeros(())
    return values.max()


# ---------------------------------------------------------------------- #
# 主接口
# ---------------------------------------------------------------------- #
def mesh_geometry_features(
    vertices: Tensor, faces: Tensor, geo_dim: int = 256
) -> Tensor:
    """提取单个 mesh 的几何描述子。

    Args:
        vertices: [V, 3]（也接受 [1, V, 3]）顶点坐标。
        faces: [F, 3]（也接受 [1, F, 3]）面索引。
        geo_dim: 输出维度，须 >= :data:`MIN_GEO_DIM`。

    Returns:
        [geo_dim] 特征向量，与 ``vertices`` 同 device / dtype，对顶点可导。
        空 mesh 返回全零向量。
    """
    bins_edge, bins_curv, bins_radial = _histogram_bins(geo_dim)

    if vertices.dim() == 3:
        vertices = vertices[0]
    if faces.dim() == 3:
        faces = faces[0]
    if vertices.dim() != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices 形状应为 [V, 3]，实际为 {tuple(vertices.shape)}")

    vertices = vertices.float()
    if vertices.numel() == 0:
        return vertices.new_zeros(geo_dim)
    faces = faces.reshape(-1, 3).long() if faces.numel() > 0 else faces.new_zeros((0, 3))

    # ---- 基础量 ---- #
    lower, upper = vertices.min(dim=0).values, vertices.max(dim=0).values
    extent = upper - lower
    diagonal = extent.norm()
    scale = diagonal.clamp(min=EPS)  # 长度类特征的统一归一化基准

    centroid = vertices.mean(dim=0)
    radial = (vertices - centroid).norm(dim=-1)
    edges = _edge_lengths(vertices, faces)
    areas = _face_areas(vertices, faces)
    curvature = _uniform_laplacian_magnitude(vertices, faces)

    edge_mean = _mean(edges)
    edge_ref = edge_mean.clamp(min=EPS)
    radial_mean = _mean(radial)

    counts = vertices.new_tensor(
        [float(vertices.shape[0]), float(faces.shape[0])]
    )
    # 排序后的包围盒边长比例：去掉坐标轴顺序的影响，只保留"扁 / 长 / 方"
    extent_sorted = torch.sort(extent, descending=True).values / scale

    scalars = torch.stack(
        [
            torch.log1p(counts[0]) / 10.0,  # 1 顶点规模
            torch.log1p(counts[1]) / 10.0,  # 2 面片规模
            extent_sorted[0],  # 3
            extent_sorted[1],  # 4 包围盒长宽高比例
            extent_sorted[2],  # 5
            diagonal / (1.0 + diagonal),  # 6 绝对尺度（异常构造常撑大包围盒）
            radial_mean / scale,  # 7
            _std(radial) / scale,  # 8 质心距离分布
            torch.tanh(_max(radial) / radial_mean.clamp(min=EPS) - 1.0),  # 9 突出程度
            edge_mean / scale,  # 10
            torch.tanh(_std(edges) / edge_ref),  # 11 边长均匀性
            torch.tanh(_max(edges) / edge_ref - 1.0),  # 12 最长边（拉伸/自交线索）
            areas.sum() / (scale**2).clamp(min=EPS),  # 13 相对表面积
            torch.tanh(_std(areas) / _mean(areas).clamp(min=EPS)),  # 14 面片均匀性
            torch.tanh(_mean(curvature) / edge_ref),  # 15
            torch.tanh(_std(curvature) / edge_ref),  # 16 曲率分布
            (centroid - 0.5 * (lower + upper)).norm() / scale,  # 17 质心偏移（对称性）
        ]
    )

    histograms = torch.cat(
        [
            # 边长按平均边长归一化：正常 mesh 集中在 1 附近，异常拉伸产生长尾
            _soft_histogram(edges / edge_ref, 0.0, 3.0, bins_edge),
            # 曲率同样以平均边长为尺度：尖刺集中在高值 bin
            _soft_histogram(curvature / edge_ref, 0.0, 2.0, bins_curv),
            # 径向距离按最远点归一化：多头/多腿会在中高段堆出额外的峰
            _soft_histogram(radial / _max(radial).clamp(min=EPS), 0.0, 1.0, bins_radial),
        ],
        dim=0,
    )

    features = torch.cat([scalars, histograms], dim=0)
    # 退化 mesh（零面积、重合顶点）可能产出 nan/inf，兜底为 0，避免污染 Critic
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def batch_geometry_features(
    vertices: Union[Tensor, Sequence[Tensor]],
    faces: Union[Tensor, Sequence[Tensor]],
    geo_dim: int = 256,
) -> Tensor:
    """批量提取几何描述子。

    Args:
        vertices: [B, V, 3] 张量（batch 共享拓扑）或长度 B 的 [V_i, 3] 列表。
        faces: [B, F, 3] / [F, 3] 张量，或长度 B（或 1，表示共享）的列表。
        geo_dim: 输出维度。

    Returns:
        [B, geo_dim] 特征矩阵。逐样本计算后堆叠——不同异常 mesh 的顶点数
        不一致，无法向量化。
    """
    vertices_list: List[Tensor] = (
        [vertices[i] for i in range(vertices.shape[0])]
        if isinstance(vertices, Tensor)
        else list(vertices)
    )
    if not vertices_list:
        raise ValueError("vertices 为空，无法提取几何特征")

    if isinstance(faces, Tensor):
        faces_list: List[Tensor] = (
            [faces] if faces.dim() == 2 else [faces[i] for i in range(faces.shape[0])]
        )
    else:
        faces_list = list(faces)
    if len(faces_list) == 1:
        faces_list = faces_list * len(vertices_list)
    if len(faces_list) != len(vertices_list):
        raise ValueError(
            f"vertices ({len(vertices_list)}) 与 faces ({len(faces_list)}) 数量不一致"
        )

    return torch.stack(
        [
            mesh_geometry_features(vertex, face, geo_dim=geo_dim)
            for vertex, face in zip(vertices_list, faces_list)
        ],
        dim=0,
    )

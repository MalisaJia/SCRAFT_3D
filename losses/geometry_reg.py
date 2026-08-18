"""Mesh 几何正则化损失。

纯 PyTorch 实现（不依赖 pytorch3d / kaolin），支持
``vertices`` 为 [V, 3] 或 [B, V, 3]，``faces`` 为 [F, 3] 三角面片索引
（batch 内共享同一拓扑，这也是模板形变式 mesh 生成器的常见设定）。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-8


def _canonicalize(vertices: Tensor, faces: Tensor) -> Tuple[Tensor, Tensor]:
    """把输入统一成 vertices=[B, V, 3]、faces=[F, 3]。"""
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    if vertices.dim() != 3 or vertices.size(-1) != 3:
        raise ValueError(
            f"vertices 应为 [V, 3] 或 [B, V, 3]，得到 {tuple(vertices.shape)}"
        )
    if faces.dim() == 3:
        # batch 内拓扑一致，取第一个即可
        faces = faces[0]
    if faces.dim() != 2 or faces.size(-1) != 3:
        raise ValueError(f"faces 应为 [F, 3] 三角面片，得到 {tuple(faces.shape)}")
    return vertices, faces.long()


def _undirected_edges(faces: Tensor) -> Tensor:
    """由面片抽取去重后的无向边，返回 [E, 2]（列内已升序）。"""
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )
    edges, _ = edges.sort(dim=1)
    return torch.unique(edges, dim=0)


def _face_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """计算单位面法线，返回 [B, F, 3]。"""
    v0 = vertices[:, faces[:, 0]]
    v1 = vertices[:, faces[:, 1]]
    v2 = vertices[:, faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return F.normalize(normals, dim=-1, eps=EPS)


def _adjacent_face_pairs(faces: Tensor) -> Tuple[Tensor, Tensor]:
    """找出共享同一条边的面片对，返回两组面索引 (f1, f2)。"""
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )
    edges, _ = edges.sort(dim=1)
    face_idx = torch.arange(faces.size(0), device=faces.device).repeat(3)

    _, inverse = torch.unique(edges, dim=0, return_inverse=True)
    order = torch.argsort(inverse)
    sorted_edge_id = inverse[order]
    sorted_face = face_idx[order]

    # 相邻位置属于同一条边 => 这两个面共享该边
    shared = sorted_edge_id[1:] == sorted_edge_id[:-1]
    return sorted_face[:-1][shared], sorted_face[1:][shared]


class GeometryRegularization(nn.Module):
    """Mesh几何正则化。

    包含：
    1. 平滑度损失：相邻顶点的法线应相似（Laplacian smoothing）
    2. 自交惩罚：mesh面片不应自我相交
    3. 边长正则化：防止过长或过短的边

    注：严格的自交检测需要 O(F^2) 的三角形相交测试，训练中不可承受。
    这里用"相邻面法线反向折叠"（dihedral 角接近 180°）作为廉价代理，
    由 ``fold_weight`` 控制，默认关闭（0.0）以免干扰早期训练。

    Args:
        smoothness_weight: Laplacian 平滑项权重
        edge_weight: 边长正则项权重
        normal_consistency_weight: 法线一致性项权重
        fold_weight: 折叠/自交代理项权重，默认 0.0（关闭）
        target_edge_length: 目标边长；None 时使用当前 mesh 的平均边长
    """

    def __init__(
        self,
        smoothness_weight: float = 1.0,
        edge_weight: float = 0.5,
        normal_consistency_weight: float = 0.5,
        fold_weight: float = 0.0,
        target_edge_length: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.edge_weight = edge_weight
        self.normal_consistency_weight = normal_consistency_weight
        self.fold_weight = fold_weight
        self.target_edge_length = target_edge_length

    def laplacian_smoothing(self, vertices: Tensor, faces: Tensor) -> Tensor:
        """拉普拉斯平滑损失：每个顶点与其邻居顶点的差异。

        Args:
            vertices: [V, 3] 或 [B, V, 3]
            faces: [F, 3]
        Returns:
            loss: scalar，uniform Laplacian 算子输出的平方模长均值
        """
        verts, faces = _canonicalize(vertices, faces)
        batch, num_verts, _ = verts.shape
        edges = _undirected_edges(faces)

        # 双向累加邻居坐标与度数
        src = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
        dst = torch.cat([edges[:, 1], edges[:, 0]], dim=0)

        neighbor_sum = torch.zeros_like(verts)
        neighbor_sum.index_add_(1, src, verts[:, dst])

        degree = torch.zeros(num_verts, device=verts.device, dtype=verts.dtype)
        degree.index_add_(0, src, torch.ones_like(src, dtype=verts.dtype))
        valid = degree > 0
        safe_degree = degree.clamp(min=1.0).view(1, num_verts, 1)

        laplacian = verts - neighbor_sum / safe_degree
        sq_norm = laplacian.pow(2).sum(dim=-1)
        # 孤立顶点没有邻居，不参与统计
        return sq_norm[:, valid].mean() if valid.any() else verts.sum() * 0.0

    def edge_length_regularization(self, vertices: Tensor, faces: Tensor) -> Tensor:
        """边长正则化：惩罚过长的边。

        以目标边长的平方做归一化，使损失对 mesh 整体尺度不敏感。

        Args:
            vertices: [V, 3] 或 [B, V, 3]
            faces: [F, 3]
        Returns:
            loss: scalar
        """
        verts, faces = _canonicalize(vertices, faces)
        edges = _undirected_edges(faces)

        lengths = (verts[:, edges[:, 0]] - verts[:, edges[:, 1]]).norm(dim=-1)
        if self.target_edge_length is None:
            # 用 detach 的平均边长作为目标，只惩罚长度的不均匀性
            target = lengths.mean(dim=1, keepdim=True).detach().clamp(min=EPS)
        else:
            target = torch.as_tensor(
                self.target_edge_length, device=verts.device, dtype=verts.dtype
            ).clamp(min=EPS)
        return ((lengths - target) / target).pow(2).mean()

    def normal_consistency(self, vertices: Tensor, faces: Tensor) -> Tensor:
        """法线一致性：相邻面的法线应相似。

        Args:
            vertices: [V, 3] 或 [B, V, 3]
            faces: [F, 3]
        Returns:
            loss: scalar，取值范围 [0, 2]（0 表示完全共面）
        """
        verts, faces = _canonicalize(vertices, faces)
        f1, f2 = _adjacent_face_pairs(faces)
        if f1.numel() == 0:
            return verts.sum() * 0.0

        normals = _face_normals(verts, faces)
        cos = (normals[:, f1] * normals[:, f2]).sum(dim=-1)
        return (1.0 - cos).mean()

    def fold_penalty(self, vertices: Tensor, faces: Tensor) -> Tensor:
        """自交代理项：惩罚相邻面法线反向（面片折叠穿插）。

        Args:
            vertices: [V, 3] 或 [B, V, 3]
            faces: [F, 3]
        Returns:
            loss: scalar，仅在相邻法线夹角 > 90° 时非零
        """
        verts, faces = _canonicalize(vertices, faces)
        f1, f2 = _adjacent_face_pairs(faces)
        if f1.numel() == 0:
            return verts.sum() * 0.0

        normals = _face_normals(verts, faces)
        cos = (normals[:, f1] * normals[:, f2]).sum(dim=-1)
        return F.relu(-cos).mean()

    def forward(self, vertices: Tensor, faces: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            vertices: [V, 3] 或 [B, V, 3]
            faces: [F, 3] 或 [B, F, 3]
        Returns:
            dict with 'smoothness', 'edge_reg', 'normal_consistency', 'total'
            （``fold_weight > 0`` 时额外含 'fold_penalty'）
        """
        smoothness = self.laplacian_smoothing(vertices, faces)
        edge_reg = self.edge_length_regularization(vertices, faces)
        normal_cons = self.normal_consistency(vertices, faces)

        total = (
            self.smoothness_weight * smoothness
            + self.edge_weight * edge_reg
            + self.normal_consistency_weight * normal_cons
        )
        out = {
            "smoothness": smoothness,
            "edge_reg": edge_reg,
            "normal_consistency": normal_cons,
        }
        if self.fold_weight > 0:
            fold = self.fold_penalty(vertices, faces)
            total = total + self.fold_weight * fold
            out["fold_penalty"] = fold
        out["total"] = total
        return out

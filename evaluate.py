"""CRaFT-3D 量化评估脚本。

计算三类指标：

1. **CLIP Score**：生成 mesh -> 多视角渲染 -> ``CLIPEncoder.encode_images``，
   与数据集对应文本（Objaverse 的 caption / ShapeNet 的类别名）的
   ``encode_text`` 结果做逐视角余弦相似度，再取均值。
2. **FID**：生成图像集合 vs 真实 mesh 渲染图像集合，用预训练 InceptionV3
   的 2048 维 pool 特征计算 Fréchet 距离。torchvision 不可用时跳过并警告。
   注意：本实现使用 torchvision 权重（非 TF-ported Inception），数值可与
   论文常用实现有系统性偏差，适合同一套代码内的相对比较。
3. **Geometry Quality**：相邻面法线一致性（复用 ``losses/geometry_reg.py``
   的邻接面 / 法线计算）与采样式自交检测比率。

用法：
    python evaluate.py --config configs/diffusion_inference.yaml \
        --dataset objaverse --data_root ./data/objaverse --num_samples 16 \
        --batch_size 4 --output report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

# 允许从任意工作目录运行本脚本
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.objaverse import ObjaverseDataset  # noqa: E402
from datasets.shapenet import ShapeNetDataset  # noqa: E402
from inference import (  # noqa: E402
    build_clip,
    build_diffusion_generator,
    build_renderer,
    export_obj,
    load_yaml_config,
    resolve_device,
    search_seed,
)
# 直接复用几何正则里的邻接面 / 面法线计算，保证与训练期指标定义一致
from losses.geometry_reg import (  # noqa: E402
    _adjacent_face_pairs,
    _canonicalize,
    _face_normals,
)

try:
    import torchvision

    _TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    torchvision = None
    _TORCHVISION_AVAILABLE = False

# ImageNet 归一化参数（InceptionV3 特征提取用）
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ====================================================================== #
# mesh batch 处理与真实样本渲染
# ====================================================================== #
def mesh_collate(samples):
    """把 mesh 样本打成 batch。"""
    valid = [s for s in samples if s["vertices"].numel() > 0 and s["faces"].numel() > 0]
    if not valid:
        return None
    batch = {
        "vertices": [s["vertices"] for s in valid],
        "faces": [s["faces"] for s in valid],
        "label": torch.stack([s["label"] for s in valid]),
        "category": [s["category"] for s in valid],
    }
    if "text" in valid[0]:
        batch["text"] = [s["text"] for s in valid]
    if "uid" in valid[0]:
        batch["uid"] = [s["uid"] for s in valid]
    return batch


def render_mesh_list(renderer, vertices_list, faces_list, camera_poses, device):
    """渲染一批拓扑各异的 mesh。"""
    shared_faces = len(faces_list) == 1 and len(vertices_list) > 1
    images = []
    for index, vertices in enumerate(vertices_list):
        faces = faces_list[0] if shared_faces else faces_list[index]
        out = renderer.render(
            vertices.to(device).unsqueeze(0),
            faces.to(device),
            camera_poses=camera_poses[index : index + 1],
        )
        images.append(out["images"][0])
    return torch.stack(images, dim=0)


def render_real_batch(renderer, vertices_list, faces_list, camera_poses, device):
    """渲染一个真实 mesh batch。"""
    return render_mesh_list(renderer, vertices_list, faces_list, camera_poses, device)


def _safe_mean(values: Sequence[float]) -> float:
    """忽略 NaN / inf 的均值；无有效值时返回 NaN。

    退化 mesh（孤立顶点 / 面数不足）的部分指标无法定义，不能拿 0 参与平均。
    """
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else float("nan")


def _json_number(value: float, digits: int = 6) -> Optional[float]:
    """把指标整理成可序列化的数值（NaN / inf -> None，避免非法 JSON）。"""
    return round(float(value), digits) if math.isfinite(float(value)) else None


# ====================================================================== #
# 可选外部打分器：Reward3D（懒加载，缺依赖自动跳过）
# ====================================================================== #
def load_reward3d_scorer(repo_path: str, device: torch.device) -> Optional[Any]:
    """懒加载外部 Reward3D 打分器。

    ``--reward3d_repo`` 指向 Reward3D 仓库本地路径；导入 / 构造失败时打印
    提示并返回 None（评估其余部分不受影响，reward3d 列整体跳过）。
    """
    abs_path = os.path.abspath(repo_path)
    if not os.path.isdir(abs_path):
        print(f"[警告] --reward3d_repo 路径不存在，reward3d 已跳过: {abs_path}")
        return None
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)
    try:
        import reward3d  # type: ignore  # 外部仓库，懒加载
    except ImportError as exc:
        print(
            f"[警告] Reward3D 依赖不可用（{exc}），reward3d 已跳过；"
            f"请按 Reward3D 仓库 README 安装其依赖后重试"
        )
        return None
    try:
        scorer_cls = getattr(reward3d, "Scorer", None) or getattr(
            reward3d, "Reward3D", None
        )
        if scorer_cls is None:
            raise AttributeError("Reward3D 仓库未暴露 Scorer / Reward3D 入口类")
        scorer = scorer_cls(device=str(device))
        print(f"Reward3D 打分器已加载: {abs_path}")
        return scorer
    except Exception as exc:  # 权重缺失 / 构造失败等，一律降级跳过
        print(f"[警告] Reward3D 初始化失败（{exc}），reward3d 已跳过")
        return None


def reward3d_score_mesh(
    scorer: Any, vertices: Tensor, faces: Tensor, tmp_dir: str, index: int
) -> Optional[float]:
    """用 Reward3D 打分器给单个 mesh 打分；失败返回 None。

    外部 API 不固定，依次尝试 ``score_mesh(path)`` / ``score(path)`` /
    ``__call__(path)``；mesh 先导出为临时 OBJ。
    """
    tris = faces[0] if faces.dim() == 3 else faces
    verts = vertices[0] if vertices.dim() == 3 else vertices
    tmp_path = os.path.join(tmp_dir, f"reward3d_tmp_{index}.obj")
    try:
        export_obj(tmp_path, verts.detach().cpu(), tris.detach().cpu())
        for name in ("score_mesh", "score", "__call__"):
            fn = getattr(scorer, name, None)
            if not callable(fn):
                continue
            try:
                return float(fn(tmp_path))
            except TypeError:
                # 部分实现直接吃 numpy 顶点 / 面片
                return float(fn(verts.detach().cpu().numpy(), tris.detach().cpu().numpy()))
        print("[警告] Reward3D 打分器未提供可用的打分接口，reward3d 已跳过")
        return None
    except Exception as exc:
        print(f"[警告] Reward3D 单样本打分失败（{exc}），该样本记为 None")
        return None


# ====================================================================== #
# 数据
# ====================================================================== #
def build_dataset(
    name: str,
    data_root: str,
    split: str = "test",
    categories: Optional[List[str]] = None,
    annotation_file: Optional[str] = None,
) -> Any:
    """构建评估用数据集（关闭增强，保证真实分布不被随机旋转扰动）。

    Raises:
        ValueError: 未知的数据集名称。
        RuntimeError: 数据集为空。
    """
    key = name.lower()
    if key == "shapenet":
        dataset: Any = ShapeNetDataset(
            data_root=data_root,
            split=split,
            categories=categories,
            augment=False,
            num_points=None,  # 重采样会破坏 faces 索引，渲染路径下必须关闭
        )
    elif key == "objaverse":
        dataset = ObjaverseDataset(
            data_root=data_root,
            annotation_file=annotation_file,
            lvis_categories=categories,
            num_points=None,
        )
    else:
        raise ValueError(f"未知的 dataset: {name!r}，可选 'shapenet' / 'objaverse'")

    if len(dataset) == 0:
        raise RuntimeError(
            f"数据集 {name!r} 在 {data_root!r} 下没有样本，请检查 --data_root / --categories"
        )
    return dataset


def batch_prompts(batch: Dict[str, Any]) -> List[str]:
    """取出每个样本用于 CLIP 打分的文本（Objaverse caption 优先，其次类别名）。"""
    texts = batch.get("text")
    categories = list(batch.get("category", []))
    prompts: List[str] = []
    for index in range(len(batch["vertices"])):
        if texts is not None and index < len(texts) and texts[index]:
            prompts.append(str(texts[index]))
        elif index < len(categories) and categories[index]:
            prompts.append(f"a {categories[index]}")
        else:
            prompts.append("a 3d object")
    return prompts


def trim_batch(batch: Dict[str, Any], size: int) -> Dict[str, Any]:
    """把 batch 截断到指定样本数（最后一个 batch 用于精确凑满 num_samples）。"""
    if size >= len(batch["vertices"]):
        return batch
    trimmed: Dict[str, Any] = {}
    for key, value in batch.items():
        trimmed[key] = value[:size] if isinstance(value, (list, Tensor)) else value
    return trimmed


# ====================================================================== #
# 指标 1：CLIP Score
# ====================================================================== #
def clip_score_batch(clip: Any, images: Tensor, prompts: List[str]) -> Tensor:
    """计算一个 batch 渲染图与其文本的逐视角 CLIP 相似度。

    Args:
        clip: CLIPEncoder 实例。
        images: [B, N_views, 3, H, W]，范围 [0, 1]。
        prompts: 长度 B 的文本列表。

    Returns:
        [B * N_views] 余弦相似度（无梯度）。
    """
    batch_size, num_views = images.shape[:2]
    if len(prompts) != batch_size:
        raise ValueError(f"prompts 数量 {len(prompts)} 与 batch 大小 {batch_size} 不一致")

    with torch.no_grad():
        text_features = clip.encode_text(prompts)  # [B, D]
        # 每个样本的文本重复到它的所有视角上，再做一一配对
        text_features = text_features.unsqueeze(1).expand(-1, num_views, -1).flatten(0, 1)
        image_features = clip.encode_images(images.flatten(0, 1))  # [B*N, D]
        return clip.paired_similarity(image_features, text_features)


# ====================================================================== #
# 指标 2：FID
# ====================================================================== #
class InceptionFeatureExtractor:
    """预训练 InceptionV3 的 2048 维 pool 特征提取器（用于 FID）。"""

    def __init__(self, device: torch.device) -> None:
        if not _TORCHVISION_AVAILABLE:
            raise RuntimeError("计算 FID 需要 torchvision，请先 pip install torchvision")

        from torchvision.models import inception_v3

        try:  # torchvision >= 0.13 的 weights 枚举 API
            from torchvision.models import Inception_V3_Weights

            model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
            )
        except ImportError:  # pragma: no cover - 老版本 torchvision
            model = inception_v3(pretrained=True, transform_input=False)

        model.fc = torch.nn.Identity()  # 输出 2048 维 pool 特征
        self.model = model.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.device = device
        self._mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images: Tensor, chunk_size: int = 32) -> Tensor:
        """提取特征。

        Args:
            images: [B, 3, H, W] 或 [B, N, 3, H, W]，范围 [0, 1]。
            chunk_size: 分块前向的大小（控制显存占用）。

        Returns:
            [B(*N), 2048] float32 特征（CPU 上返回，便于长期累积）。
        """
        if images.dim() == 5:
            images = images.flatten(0, 1)
        images = images.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)

        features: List[Tensor] = []
        for start in range(0, images.shape[0], chunk_size):
            chunk = images[start : start + chunk_size]
            chunk = torch.nn.functional.interpolate(
                chunk, size=(299, 299), mode="bilinear", align_corners=False
            )
            chunk = (chunk - self._mean) / self._std
            features.append(self.model(chunk).float().cpu())
        return torch.cat(features, dim=0)


def _sqrtm_psd(matrix: Tensor) -> Tensor:
    """对称半正定矩阵的平方根（特征分解实现，免去 scipy 依赖）。"""
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp(min=0.0)
    return (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.transpose(-1, -2)


def compute_fid(real_features: Tensor, fake_features: Tensor, eps: float = 1e-6) -> float:
    """计算两组特征之间的 Fréchet Inception Distance。

    ``tr((Σ1 Σ2)^{1/2})`` 用对称化技巧求解：
    ``tr(sqrtm(Σ1 Σ2)) = Σ sqrt(eig(A Σ2 A))``，其中 ``A = sqrtm(Σ1)``，
    ``A Σ2 A`` 对称半正定，因此只需要 ``eigh``，不依赖 scipy。

    Args:
        real_features: [N1, D]
        fake_features: [N2, D]
        eps: 协方差对角扰动，提升数值稳定性。

    Returns:
        FID 标量（越小越好）。
    """
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise ValueError("计算 FID 至少需要 2 个样本")
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("两组特征的维度必须一致")

    real = real_features.double()
    fake = fake_features.double()
    mu_real, mu_fake = real.mean(dim=0), fake.mean(dim=0)

    identity = torch.eye(real.shape[1], dtype=torch.float64)
    sigma_real = torch.cov(real.t()) + eps * identity
    sigma_fake = torch.cov(fake.t()) + eps * identity

    sqrt_real = _sqrtm_psd(sigma_real)
    middle = sqrt_real @ sigma_fake @ sqrt_real
    middle = 0.5 * (middle + middle.t())
    trace_sqrt = torch.linalg.eigvalsh(middle).clamp(min=0.0).sqrt().sum()

    diff = (mu_real - mu_fake).pow(2).sum()
    fid = diff + torch.trace(sigma_real) + torch.trace(sigma_fake) - 2.0 * trace_sqrt
    return float(max(fid.item(), 0.0))


# ====================================================================== #
# 指标 3：几何质量
# ====================================================================== #
def adjacent_normal_cosines(vertices: Tensor, faces: Tensor) -> Tensor:
    """每对相邻面的法线余弦。

    完全复用 ``losses/geometry_reg.py`` 的邻接面搜索与面法线计算，
    保证评估指标与训练时的 ``normal_consistency`` 定义一致。

    Args:
        vertices: [V, 3] 或 [B, V, 3]
        faces: [F, 3] 或 [B, F, 3]

    Returns:
        [B, P] 余弦（已 clamp 到 [-1, 1]）；无邻接面时返回空张量。
    """
    verts, tris = _canonicalize(vertices, faces)
    f1, f2 = _adjacent_face_pairs(tris)
    if f1.numel() == 0:
        return verts.new_zeros((verts.shape[0], 0))
    normals = _face_normals(verts, tris)
    return (normals[:, f1] * normals[:, f2]).sum(dim=-1).clamp(-1.0, 1.0)


def normal_consistency_score(vertices: Tensor, faces: Tensor) -> Tuple[float, float]:
    """相邻面法线一致性。

    Args:
        vertices: [V, 3] 或 [B, V, 3]
        faces: [F, 3] 或 [B, F, 3]

    Returns:
        (mean_cos，mean_angle_degrees)：平均余弦（越接近 1 越光滑）与平均夹角。
    """
    return _cosine_stats(adjacent_normal_cosines(vertices, faces))[:2]


def _cosine_stats(cos: Tensor) -> Tuple[float, float, float]:
    """由相邻面余弦得出 (平均余弦, 平均夹角度数, 反向夹角占比)。"""
    if cos.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    angle = torch.rad2deg(torch.acos(cos))
    return (
        float(cos.mean().item()),
        float(angle.mean().item()),
        float((cos < 0).float().mean().item()),
    )


def folded_pair_ratio(vertices: Tensor, faces: Tensor) -> float:
    """相邻面法线反向（夹角 > 90°）的比例，作为面片折叠的廉价指标。"""
    return _cosine_stats(adjacent_normal_cosines(vertices, faces))[2]


def _segment_triangle_hit(
    origin: Tensor,
    end: Tensor,
    v0: Tensor,
    v1: Tensor,
    v2: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """向量化的线段-三角形相交测试（Möller–Trumbore）。

    为避免把"仅顶点/边接触"误判为自交，重心坐标与线段参数都使用开区间。

    Args:
        origin/end: [P, 3] 线段端点。
        v0/v1/v2: [P, 3] 三角形顶点。

    Returns:
        [P] bool，True 表示线段穿过三角形内部。
    """
    direction = end - origin
    edge1, edge2 = v1 - v0, v2 - v0

    pvec = torch.cross(direction, edge2, dim=-1)
    det = (edge1 * pvec).sum(-1)
    parallel = det.abs() < eps
    inv_det = 1.0 / torch.where(parallel, torch.ones_like(det), det)

    tvec = origin - v0
    u = (tvec * pvec).sum(-1) * inv_det
    qvec = torch.cross(tvec, edge1, dim=-1)
    v = (direction * qvec).sum(-1) * inv_det
    t = (edge2 * qvec).sum(-1) * inv_det

    inside = (u > eps) & (v > eps) & (u + v < 1.0 - eps)
    within = (t > eps) & (t < 1.0 - eps)
    return (~parallel) & inside & within


def self_intersection_ratio(
    vertices: Tensor,
    faces: Tensor,
    num_pairs: int = 20000,
    generator: Optional[torch.Generator] = None,
) -> Tuple[float, float]:
    """采样式自交检测。

    精确自交检测是 O(F^2) 的三角形对测试（2048 顶点的模板约 4k 面 -> 8M 对），
    这里随机采样面片对并做精确的三角形-三角形相交测试（转化为 6 次线段-三角形
    测试），先用 AABB 快速剔除。共享顶点的相邻面天然接触，不计入。

    Args:
        vertices: [V, 3]（单个 mesh）
        faces: [F, 3]
        num_pairs: 采样的面片对数量。
        generator: 可选随机数生成器，便于复现。

    Returns:
        (相交面片对占已测试面片对的比例，涉及自交的面片占全部面片的比例)
    """
    verts, tris = _canonicalize(vertices, faces)
    verts = verts[0]
    num_faces = tris.shape[0]
    if num_faces < 2:
        return float("nan"), float("nan")

    device = verts.device
    num_pairs = max(1, int(num_pairs))
    idx_a = torch.randint(num_faces, (num_pairs,), device=device, generator=generator)
    idx_b = torch.randint(num_faces, (num_pairs,), device=device, generator=generator)

    tri_a, tri_b = tris[idx_a], tris[idx_b]  # [P, 3] 顶点索引
    # 共享任意顶点的面片（自身 / 邻面）排除在外
    shares_vertex = (tri_a.unsqueeze(-1) == tri_b.unsqueeze(-2)).any(dim=-1).any(dim=-1)
    valid = ~shares_vertex
    if not bool(valid.any()):
        return float("nan"), float("nan")

    idx_a, idx_b = idx_a[valid], idx_b[valid]
    pa, pb = verts[tris[idx_a]], verts[tris[idx_b]]  # [P, 3, 3]

    # AABB 预筛：包围盒不相交则一定不相交
    overlap = (
        (pa.min(dim=1).values <= pb.max(dim=1).values)
        & (pb.min(dim=1).values <= pa.max(dim=1).values)
    ).all(dim=-1)
    tested = int(idx_a.numel())
    hits = torch.zeros(tested, dtype=torch.bool, device=device)

    if bool(overlap.any()):
        ca, cb = pa[overlap], pb[overlap]
        candidate_hits = torch.zeros(ca.shape[0], dtype=torch.bool, device=device)
        # A 的三条边穿过 B，以及 B 的三条边穿过 A
        for src, dst in ((ca, cb), (cb, ca)):
            for i in range(3):
                candidate_hits |= _segment_triangle_hit(
                    src[:, i], src[:, (i + 1) % 3], dst[:, 0], dst[:, 1], dst[:, 2]
                )
        hits[overlap] = candidate_hits

    pair_ratio = float(hits.float().mean().item())
    involved = torch.cat([idx_a[hits], idx_b[hits]])
    face_ratio = float(torch.unique(involved).numel() / num_faces) if tested else float("nan")
    return pair_ratio, face_ratio


def geometry_metrics(
    vertices: Tensor,
    faces: Tensor,
    num_pairs: int = 20000,
) -> Dict[str, float]:
    """对一个 batch 的 mesh 计算几何质量指标（逐样本求平均）。

    Args:
        vertices: [B, V, 3]
        faces: [F, 3] 或 [B, F, 3]
        num_pairs: 自交检测采样的面片对数量。

    Returns:
        含 normal_consistency / normal_angle_deg / self_intersection_ratio /
        intersecting_face_ratio / folded_pair_ratio 的字典。
    """
    tris = faces[0] if faces.dim() == 3 else faces
    records: Dict[str, List[float]] = {}

    for index in range(vertices.shape[0]):
        single = vertices[index : index + 1]
        # 邻接面搜索带 torch.unique，每个样本只做一次，三个法线指标共用结果
        cos_mean, angle_mean, folded = _cosine_stats(
            adjacent_normal_cosines(single, tris)
        )
        pair_ratio, face_ratio = self_intersection_ratio(single, tris, num_pairs)
        sample = {
            "normal_consistency": cos_mean,
            "normal_angle_deg": angle_mean,
            "self_intersection_ratio": pair_ratio,
            "intersecting_face_ratio": face_ratio,
            "folded_pair_ratio": folded,
        }
        for key, value in sample.items():
            records.setdefault(key, []).append(value)

    return {
        key: _safe_mean(values) for key, values in records.items()
    }


# ====================================================================== #
# 后端感知的生成入口（diffusion）
# ====================================================================== #
def generate_meshes_for_eval(
    generator: Any,
    renderer: Any,
    clip: Any,
    config: Dict[str, Any],
    num_samples: int,
    captions: Optional[List[str]] = None,
    num_candidates: int = 1,
) -> List[Dict[str, Tensor]]:
    """为评估生成 mesh（diffusion 后端）。

    基于种子采样逐样本生成；``num_candidates > 1`` 且 CLIP 可用时对数据集
    caption 做 CLIP 种子重排序（与 inference.py ``search_seed`` 一致）。

    Args:
        generator: ``DiffusionMeshGenerator`` 实例。
        renderer: 多视角渲染器（种子重排序打分用）。
        clip: CLIPEncoder；None 时不做重排序。
        config: diffusion 推理配置（候选数缺省读 ``inference.num_candidates``）。
        num_samples: 需要生成的 mesh 数量。
        captions: 长度 >= num_samples 的文本列表（Objaverse 逐样本 caption）。
        num_candidates: 种子重排序候选数；<= 1 时单种子直出。

    Returns:
        长度 num_samples 的列表，每项 ``{'vertices': [1, V, 3], 'faces': [F, 3]}``。
        marching cubes 输出的拓扑逐样本不同，无法拼成统一 batch。
    """
    meshes: List[Dict[str, Tensor]] = []
    for index in range(int(num_samples)):
        caption = (
            captions[index]
            if captions is not None and index < len(captions) and captions[index]
            else "a 3d object"
        )
        if clip is not None and num_candidates > 1:
            result = search_seed(
                generator, renderer, clip, caption, config, num_candidates
            )
        else:
            # 单种子直出：种子由全局 RNG 派生（evaluate 入口已固定 --seed）
            cand_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            result = generator.generate(prompt=caption, seed=cand_seed)
        meshes.append({"vertices": result["vertices"], "faces": result["faces"]})
    return meshes


# ====================================================================== #
# 评估主流程
# ====================================================================== #
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """按 CLI 参数执行完整评估，返回指标报告字典。"""
    started = time.time()
    warnings: List[str] = []
    device = resolve_device(args.device)
    print(f"评估设备: {device}")

    # ---------------- 模型（diffusion 后端）---------------- #
    print("评估后端: diffusion")

    # 无 checkpoint：模型结构与采样参数全部来自独立 yaml 配置
    if args.config is None:
        raise ValueError("需要通过 --config 提供 diffusion 推理配置")
    diffusion_config: Dict[str, Any] = load_yaml_config(args.config)
    generator_diff = build_diffusion_generator(diffusion_config, device)
    config: Dict[str, Any] = diffusion_config
    renderer = build_renderer(config, device, args.num_views, args.image_size)
    # 默认 "fixed"（等间距环绕 + 中位仰角）：真实 / 生成两边视角完全一致，
    # FID 不会被相机采样差异污染，且结果可复现
    renderer.azimuth_strategy = args.camera_strategy
    num_views = renderer.num_views

    clip = None if args.skip_clip else build_clip(config, device)
    if clip is None and not args.skip_clip:
        warnings.append("CLIP 不可用，clip_score 已跳过")

    inception: Optional[InceptionFeatureExtractor] = None
    if not args.skip_fid:
        try:
            inception = InceptionFeatureExtractor(device)
        except (RuntimeError, OSError) as exc:  # 缺 torchvision / 权重下载失败
            print(f"[警告] FID 已跳过: {exc}")
            warnings.append(f"FID 已跳过: {exc}")

    # ---------------- 可选 Reward3D 外部打分器（懒加载）---------------- #
    reward3d_scorer: Optional[Any] = None
    if args.reward3d_repo:
        reward3d_scorer = load_reward3d_scorer(args.reward3d_repo, device)
        if reward3d_scorer is None:
            warnings.append("Reward3D 打分器不可用，reward3d 列已跳过")

    # ---------------- 数据 ---------------- #
    dataset = build_dataset(
        args.dataset,
        args.data_root,
        split=args.split,
        categories=args.categories or None,
        annotation_file=args.annotation_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=mesh_collate,
        drop_last=False,
    )
    print(f"数据集 {args.dataset}: {len(dataset)} 个样本，评估 {args.num_samples} 个")
    if args.num_samples > len(dataset):
        # 数据集不够时会重复遍历，真实集存在重复样本，FID 会偏低
        warnings.append(
            f"num_samples={args.num_samples} 大于数据集大小 {len(dataset)}，"
            "真实样本将被重复使用"
        )

    # ---------------- 累积器 ---------------- #
    clip_scores: List[float] = []
    geometry_records: Dict[str, List[float]] = {}
    real_features: List[Tensor] = []
    fake_features: List[Tensor] = []
    reward3d_scores: List[Optional[float]] = []
    processed = 0
    render_available = True

    torch.manual_seed(int(args.seed))
    while processed < args.num_samples:
        exhausted = True
        for batch in loader:
            if processed >= args.num_samples:
                break
            exhausted = False
            batch = trim_batch(batch, args.num_samples - processed)
            batch_size = len(batch["vertices"])

            # ---- 生成 ---- #
            # marching cubes 输出的拓扑逐样本不同，无法拼成统一 batch
            with torch.no_grad():
                meshes = generate_meshes_for_eval(
                    generator_diff,
                    renderer,
                    clip,
                    diffusion_config,
                    batch_size,
                    captions=batch_prompts(batch),
                    num_candidates=args.num_candidates,
                )
            per_sample: List[Tuple[Tensor, Tensor]] = [
                (m["vertices"], m["faces"]) for m in meshes
            ]

            # ---- 几何指标（不依赖渲染，任何环境都能算）---- #
            # mesh 拓扑不一致，逐样本计算几何指标
            for verts_single, faces_single in per_sample:
                for key, value in geometry_metrics(
                    verts_single, faces_single, num_pairs=args.intersection_pairs
                ).items():
                    geometry_records.setdefault(key, []).append(value)

            # ---- 可选 Reward3D 逐样本打分（仅在提供 --reward3d_repo 时启用）---- #
            if reward3d_scorer is not None:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for sample_idx, (verts_single, faces_single) in enumerate(per_sample):
                        reward3d_scores.append(
                            reward3d_score_mesh(
                                reward3d_scorer,
                                verts_single,
                                faces_single,
                                tmp_dir,
                                sample_idx,
                            )
                        )

            # ---- 渲染（真实 / 生成共用同一套相机，保证 FID 可比）---- #
            if render_available:
                try:
                    with torch.no_grad():
                        camera_poses, _, _ = renderer.generate_camera_poses(
                            batch_size, device=device
                        )
                        # 拓扑不一致，逐样本渲染后沿 batch 维拼接
                        fake_images = torch.cat(
                            [
                                renderer.render(
                                    v, f, camera_poses=camera_poses[i : i + 1]
                                )["images"]
                                for i, (v, f) in enumerate(per_sample)
                            ],
                            dim=0,
                        )
                        real_images = (
                            render_real_batch(
                                renderer,
                                batch["vertices"],
                                batch["faces"],
                                camera_poses,
                                device,
                            )
                            if inception is not None
                            else None
                        )
                except RuntimeError as exc:  # 典型原因：缺 nvdiffrast / 非 CUDA
                    render_available = False
                    print(f"[警告] 渲染不可用，clip_score 与 FID 已跳过: {exc}")
                    warnings.append(f"渲染不可用，clip_score 与 FID 已跳过: {exc}")
                else:
                    if clip is not None:
                        clip_scores.extend(
                            float(s)
                            for s in clip_score_batch(
                                clip, fake_images, batch_prompts(batch)
                            )
                        )
                    if inception is not None and real_images is not None:
                        fake_features.append(inception(fake_images))
                        real_features.append(inception(real_images))

            processed += batch_size
            print(
                f"  进度 {processed}/{args.num_samples}"
                + (f"  clip={np.mean(clip_scores):.4f}" if clip_scores else "")
            )
        if exhausted:  # 数据集比 num_samples 小且已遍历完，避免死循环
            break

    if processed == 0:
        raise RuntimeError("没有成功评估任何样本，请检查数据集内容")

    # ---------------- 汇总 ---------------- #
    fid: Optional[float] = None
    if real_features and fake_features:
        real_all = torch.cat(real_features, dim=0)
        fake_all = torch.cat(fake_features, dim=0)
        if min(real_all.shape[0], fake_all.shape[0]) < 2:
            warnings.append("图像数量不足，FID 已跳过")
        else:
            fid = compute_fid(real_all, fake_all)
            if min(real_all.shape[0], fake_all.shape[0]) < 2048:
                warnings.append(
                    f"FID 仅基于 {real_all.shape[0]} / {fake_all.shape[0]} 张图像，"
                    "样本量偏小，数值方差较大"
                )

    geometry = {
        key: _json_number(_safe_mean(values))
        for key, values in sorted(geometry_records.items())
    }
    report: Dict[str, Any] = {
        "clip_score": _json_number(_safe_mean(clip_scores)) if clip_scores else None,
        "fid": round(fid, 4) if fid is not None else None,
        "geometry": geometry,
        "reward3d": (
            {
                "mean": _json_number(_safe_mean(reward3d_scores)),
                "per_sample": [_json_number(s) if s is not None else None for s in reward3d_scores],
                "repo": os.path.abspath(args.reward3d_repo),
            }
            if reward3d_scorer is not None
            else None
        ),
        "num_samples": int(processed),
        "num_views": int(num_views),
        "image_size": int(renderer.image_size),
        "camera_strategy": args.camera_strategy,
        "num_rendered_images": int(processed * num_views) if render_available else 0,
        "dataset": {
            "name": args.dataset,
            "data_root": args.data_root,
            "split": args.split,
            "size": len(dataset),
        },
        "backend": "diffusion",
        "config": os.path.abspath(args.config) if args.config else None,
        "device": str(device),
        "seed": int(args.seed),
        "warnings": warnings,
        "elapsed_seconds": round(time.time() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"指标报告已写入: {args.output}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


# ====================================================================== #
# 入口
# ====================================================================== #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRaFT-3D 量化评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="diffusion 推理配置路径（如 configs/diffusion_inference.yaml）",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="diffusion 种子重排序候选数（>1 且 CLIP 可用时按 caption 重排序）",
    )
    parser.add_argument(
        "--dataset", type=str, default="shapenet", choices=["shapenet", "objaverse"]
    )
    parser.add_argument("--data_root", type=str, default="./data", help="数据集根目录")
    parser.add_argument("--num_samples", type=int, default=100, help="评估样本数")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动选择")
    parser.add_argument("--output", type=str, default="report.json", help="指标报告输出路径")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="类别过滤")
    parser.add_argument(
        "--annotation_file", type=str, default=None, help="Objaverse 标注文件路径"
    )
    parser.add_argument("--num_views", type=int, default=None, help="覆盖配置中的视角数")
    parser.add_argument("--image_size", type=int, default=None, help="覆盖配置中的渲染分辨率")
    parser.add_argument(
        "--camera_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "stratified", "random"],
        help="相机方位角采样策略，fixed 可复现且真假两边视角一致",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument(
        "--intersection_pairs",
        type=int,
        default=20000,
        help="自交检测采样的面片对数量（越大越准，越慢）",
    )
    parser.add_argument("--skip_fid", action="store_true", help="跳过 FID 计算")
    parser.add_argument("--skip_clip", action="store_true", help="跳过 CLIP Score 计算")
    parser.add_argument(
        "--reward3d_repo",
        type=str,
        default=None,
        help="Reward3D 仓库本地路径（可选）；提供时给每个生成 mesh 追加 reward3d 分数列，"
        "缺依赖时自动跳过；不提供则行为与原来完全一致",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        evaluate(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

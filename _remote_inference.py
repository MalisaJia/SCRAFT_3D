"""Semantic3D-GAN 推理脚本：文本 -> 3D mesh（支持 GAN / diffusion 双后端）。

后端一：GAN（默认，``backend="gan"`` 或缺省）
    MeshGenerator 本身是**无条件**生成器（``forward(z)`` 只吃潜在向量），
    文本控制通过冻结 CLIP 在推理阶段完成，分两个阶段：

    1. 候选筛选：采样 ``num_candidates`` 个 z，逐个渲染多视角图像，用
       CLIP 图文余弦相似度打分，取最高分的 z。
    2. 潜变量优化（可选，``--optimize_steps > 0``）：以该 z 为起点，沿
       ``CLIP -> 渲染图像 -> mesh 顶点`` 的可微路径做梯度上升，进一步提升相似度。

后端二：diffusion（``backend="diffusion"``，见 models/diffusion_adapter.py）
    TripoSG rectified-flow 扩散模型。TripoSG 本体无文本条件，文本控制通过
    **种子搜索**实现：采样 N 个扩散噪声种子分别生成 mesh，渲染多视图后
    重排序，取最高分结果（``search_seed``）。打分器由 ``inference.scorer``
    选择：``clip``（CLIP 图文相似度，默认）/ ``critic``（GAN 时代训练的
    SemanticCritic 结构合理性分数）/ ``combo``（两者 min-max 归一后加权）。

用法：
    # GAN 后端
    python inference.py --checkpoint outputs/ckpt_final.pt --prompt "a red chair" \
        --output_dir ./results --num_views 8 --image_size 256

    # diffusion 后端（后端类型自动从 config 的 model.backend 读取）
    python inference.py --config configs/diffusion_inference.yaml --prompt "a red chair" \
        --output_dir ./results

输出（``{output_dir}/{prompt_slug}_*``，两种后端格式一致）：
    *_mesh.obj   OBJ 格式 mesh（顶点 + 面）
    *_views.png  多视图拼图
    *_meta.json  生成参数与 CLIP 分数记录
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# 允许从任意工作目录运行本脚本
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.generator import MeshGenerator  # noqa: E402
from models.semantic_critic import SemanticCritic  # noqa: E402
from rendering.multi_view_render import MultiViewRenderer  # noqa: E402
from utils.geometry_features import batch_geometry_features  # noqa: E402
from utils.visualize import save_image, tile_images  # noqa: E402

try:  # diffusion 后端为可选依赖：适配器导入失败不影响 GAN 路径
    # （TripoSG 本体在 DiffusionMeshGenerator 实例化时才懒加载导入）
    from models.diffusion_adapter import DiffusionMeshGenerator

    _DIFFUSION_IMPORTABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    DiffusionMeshGenerator = None  # type: ignore[assignment]
    _DIFFUSION_IMPORTABLE = False

try:  # yaml 仅 diffusion 后端读取独立 config 时需要
    import yaml
except ImportError:  # pragma: no cover - 取决于运行环境
    yaml = None  # type: ignore[assignment]

try:  # CLIP 为可选依赖：缺失时退化为随机采样（不做文本引导）
    from vlm.clip_encoder import CLIPEncoder

    _CLIP_IMPORTABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    CLIPEncoder = None  # type: ignore[assignment]
    _CLIP_IMPORTABLE = False


# ====================================================================== #
# 通用工具
# ====================================================================== #
def slugify(text: str, max_length: int = 60) -> str:
    """把 prompt 转成适合做文件名的 slug。

    Args:
        text: 原始文本。
        max_length: 截断长度。

    Returns:
        只含小写字母、数字与下划线的字符串（空输入返回 "sample"）。
    """
    slug = re.sub(r"[^\w\s-]", "", text.strip().lower(), flags=re.UNICODE)
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug[:max_length] or "sample"


def export_obj(
    path: str,
    vertices: Tensor,
    faces: Tensor,
    comment: Optional[str] = None,
) -> str:
    """把 mesh 导出为 Wavefront OBJ。

    Args:
        path: 输出 ``.obj`` 路径。
        vertices: [V, 3] 或 [1, V, 3] 顶点坐标。
        faces: [F, 3] 或 [1, F, 3] 三角面索引（0 基）。
        comment: 可选注释，写在文件头部。

    Returns:
        实际写入的路径。
    """
    verts = vertices.detach().float().cpu()
    tris = faces.detach().long().cpu()
    if verts.dim() == 3:
        verts = verts[0]
    if tris.dim() == 3:
        tris = tris[0]
    if verts.dim() != 2 or verts.shape[-1] != 3:
        raise ValueError(f"vertices 应为 [V, 3]，实际为 {tuple(verts.shape)}")
    if tris.dim() != 2 or tris.shape[-1] != 3:
        raise ValueError(f"faces 应为 [F, 3]，实际为 {tuple(tris.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lines: List[str] = ["# Semantic3D-GAN generated mesh"]
    if comment:
        lines.append(f"# {comment}")
    lines.append(f"# vertices: {verts.shape[0]}, faces: {tris.shape[0]}")

    for x, y, z in verts.tolist():
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    # OBJ 面索引从 1 开始
    for i0, i1, i2 in (tris + 1).tolist():
        lines.append(f"f {i0} {i1} {i2}")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def resolve_device(name: Optional[str]) -> torch.device:
    """解析设备字符串，CUDA 不可用时自动回退到 CPU。"""
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[警告] 当前环境没有可用 CUDA，已回退到 CPU（nvdiffrast 渲染将不可用）")
        return torch.device("cpu")
    return device


def load_yaml_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件为 dict（diffusion 后端使用）。

    Raises:
        RuntimeError: 未安装 PyYAML。
        FileNotFoundError: 文件不存在。
    """
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，无法读取 yaml 配置，请先执行: pip install pyyaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return dict(config or {})


def resolve_backend(backend: str, config_path: Optional[str]) -> str:
    """确定推理后端：显式指定优先；``auto`` 时从 config 的 ``model.backend`` 探测。

    Args:
        backend: CLI ``--backend`` 值（"auto" / "gan" / "diffusion"）。
        config_path: diffusion config 路径；``auto`` 且提供时读取其 ``model.backend``。

    Returns:
        "gan" 或 "diffusion"（缺省一律 "gan"，保持旧行为）。
    """
    if backend and backend.lower() != "auto":
        return backend.lower()
    if config_path:
        config = load_yaml_config(config_path)
        detected = str((config.get("model", {}) or {}).get("backend", "diffusion")).lower()
        return "diffusion" if detected == "diffusion" else "gan"
    return "gan"


# ====================================================================== #
# checkpoint / 模型构建
# ====================================================================== #
def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """加载 checkpoint（``train.py:save_checkpoint`` 的输出格式）。

    Args:
        path: checkpoint 路径。
        device: map_location 目标设备。

    Returns:
        checkpoint 字典，至少包含 ``generator``；``config`` 缺失时为空 dict。

    Raises:
        FileNotFoundError: 文件不存在。
        KeyError: 缺少 generator 权重。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    try:  # torch>=2.6 默认 weights_only=True，而 checkpoint 里含 config（dict）
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - 老版本 torch 没有该参数
        state = torch.load(path, map_location=device)

    if not isinstance(state, dict):
        raise KeyError(f"checkpoint 格式异常（期望 dict，实际 {type(state).__name__}）")
    if "generator" not in state:
        available = ", ".join(sorted(map(str, state.keys())))
        raise KeyError(f"checkpoint 中缺少 'generator' 权重，现有键: {available}")

    state.setdefault("config", {})
    return state


def build_generator(
    config: Dict[str, Any],
    state_dict: Dict[str, Tensor],
    device: torch.device,
) -> MeshGenerator:
    """按配置构建 MeshGenerator 并载入权重（与 ``train.build_models`` 保持一致）。

    Args:
        config: checkpoint 中保存的完整训练配置。
        state_dict: generator 的 state_dict。
        device: 目标设备。

    Returns:
        eval 模式的生成器。
    """
    gen_cfg = dict(config.get("generator", {}) or {})
    vlm_cfg = dict(config.get("vlm", {}) or {})
    generator = MeshGenerator(
        latent_dim=int(gen_cfg.get("latent_dim", 512)),
        hidden_dim=int(gen_cfg.get("hidden_dim", 256)),
        num_layers=int(gen_cfg.get("num_layers", 8)),
        output_vertices=int(gen_cfg.get("output_vertices", 2048)),
        use_sdf=bool(gen_cfg.get("use_sdf", False)),
        # 旧 checkpoint 的 config 无此键 -> 默认 32，与旧结构一致；
        # v7 新 checkpoint 存 0 -> 不构建 feature_head，结构匹配
        vertex_feature_dim=int(gen_cfg.get("vertex_feature_dim", 32)),
        asymmetry_scale=float(gen_cfg.get("asymmetry_scale", 0.05)),
        # 渐进式细分：旧配置缺省为 0 / False，行为与旧版完全一致
        subdivision_levels=int(gen_cfg.get("subdivision_levels", 0)),
        use_refinement=bool(gen_cfg.get("use_refinement", False)),
        refinement_hidden_dim=int(gen_cfg.get("refinement_hidden_dim", 128)),
        clip_dim=int(gen_cfg.get("clip_dim", vlm_cfg.get("clip_dim", 512))),
    ).to(device)

    # strict=False 兼容旧 checkpoint：新增模块（如 refinement 精修 MLP）
    # 在旧权重中不存在时保持初始化值（输出层零初始化，残差恒为 0，无副作用）
    incompatible = generator.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(
            f"[警告] generator 权重缺少以下键（新模块将使用初始化值）: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        print(
            f"[警告] generator 权重包含以下多余键（已忽略）: "
            f"{incompatible.unexpected_keys}"
        )

    generator.eval()
    # 推理阶段参数全程冻结：潜变量优化时只对 z 求导，不堆积参数梯度
    for param in generator.parameters():
        param.requires_grad_(False)
    return generator


def build_diffusion_generator(
    config: Dict[str, Any],
    device: torch.device,
) -> "DiffusionMeshGenerator":
    """按 diffusion config 构建 ``DiffusionMeshGenerator``（TripoSG 适配器）。

    Args:
        config: 完整的 diffusion 推理配置（``configs/diffusion_inference.yaml``），
            其中 ``model.diffusion`` 段传入适配器。
        device: 目标设备。

    Returns:
        eval + 参数冻结、可直接用于推理的生成器。

    Raises:
        RuntimeError: 适配器导入失败或 TripoSG 未安装（含安装提示）。
    """
    if not _DIFFUSION_IMPORTABLE:
        raise RuntimeError(
            "未能导入 models.diffusion_adapter.DiffusionMeshGenerator，无法使用 diffusion 后端"
        )
    model_cfg = dict(config.get("model", {}) or {})
    diffusion_cfg = dict(model_cfg.get("diffusion", {}) or {})
    # 适配器构造内部已完成 eval() + requires_grad_(False) 冻结
    return DiffusionMeshGenerator(diffusion_cfg, device)


def build_renderer(
    config: Dict[str, Any],
    device: torch.device,
    num_views: Optional[int] = None,
    image_size: Optional[int] = None,
) -> MultiViewRenderer:
    """按配置构建渲染器，CLI 参数可覆盖视角数与分辨率。

    推理时使用 ``azimuth_strategy="fixed"``：等间距环绕视角，结果可复现，
    也更适合直接放进论文 / 汇报材料。
    """
    render_cfg = dict(config.get("rendering", {}) or {})
    elevation = tuple(render_cfg.get("elevation_range", (-30.0, 30.0)))
    return MultiViewRenderer(
        image_size=int(image_size or render_cfg.get("image_size", 256)),
        num_views=int(num_views or render_cfg.get("num_views", 4)),
        camera_distance=float(render_cfg.get("camera_distance", 2.5)),
        elevation_range=(float(elevation[0]), float(elevation[1])),
        azimuth_strategy="fixed",
        device=str(device),
    )


def build_clip(config: Dict[str, Any], device: torch.device) -> Optional[Any]:
    """构建冻结 CLIP 编码器，不可用时返回 None（跳过文本引导）。"""
    if not _CLIP_IMPORTABLE:
        print("[警告] 未能导入 vlm.clip_encoder，将跳过文本引导，仅随机采样生成")
        return None
    vlm_cfg = dict(config.get("vlm", {}) or {})
    try:
        return CLIPEncoder(
            model_name=str(vlm_cfg.get("model_name", "ViT-B/32")),
            device=str(device),
            input_range="zero_one",  # 渲染器输出已在 [0, 1]
            pretrained=str(vlm_cfg.get("pretrained", "openai")),
        )
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        print(f"[警告] CLIP 初始化失败（{exc}），将跳过文本引导")
        return None


def build_critic(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[SemanticCritic, int, Dict[str, Any]]:
    """从 GAN 时代 checkpoint 加载 SemanticCritic（种子重排的结构合理性打分器）。

    checkpoint 为 ``train.py:save_checkpoint`` 的输出格式：critic 权重位于
    ``state["critic"]``；``clip_dim / geo_dim / hidden_dim / dropout`` 优先取
    checkpoint 内保存的 ``config.semantic_critic``，保证结构与训练完全一致。
    注意：构建时必须先建模型再 load_state_dict——spectral_norm 的
    ``weight_u / weight_v`` 缓冲只有在构造时才生成。

    Args:
        checkpoint_path: GAN checkpoint 路径（如 outputs/ablation_a0_full/ckpt_final.pt）。
        device: 目标设备。

    Returns:
        (eval + 冻结的 SemanticCritic, geo_dim, 训练时的 rendering 配置段)。
        第三项用于构建与 Critic 训练预处理一致的打分渲染器
        （image_size / num_views / camera_distance / elevation_range）。

    Raises:
        FileNotFoundError / KeyError: checkpoint 缺失或不含 critic 权重。
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"critic checkpoint 不存在: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - 老版本 torch
        state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict) or "critic" not in state:
        available = ", ".join(sorted(map(str, state.keys()))) if isinstance(state, dict) else type(state).__name__
        raise KeyError(f"checkpoint 缺少 'critic' 权重，现有键: {available}")

    critic_cfg = dict((state.get("config", {}) or {}).get("semantic_critic", {}) or {})
    vlm_cfg = dict((state.get("config", {}) or {}).get("vlm", {}) or {})
    clip_dim = int(critic_cfg.get("clip_dim", vlm_cfg.get("clip_dim", 512)))
    geo_dim = int(critic_cfg.get("geo_dim", 256))

    critic = SemanticCritic(
        clip_dim=clip_dim,
        geo_dim=geo_dim,
        hidden_dim=int(critic_cfg.get("hidden_dim", 512)),
        dropout=float(critic_cfg.get("dropout", 0.1)),
    )
    critic.load_state_dict(state["critic"], strict=True)
    critic.to(device).eval()
    for param in critic.parameters():
        param.requires_grad_(False)
    render_cfg = dict((state.get("config", {}) or {}).get("rendering", {}) or {})
    return critic, geo_dim, render_cfg


def critic_score_for_mesh(
    critic: SemanticCritic,
    geo_dim: int,
    clip_encoder: Any,
    renderer: MultiViewRenderer,
    vertices: Tensor,
    faces: Tensor,
) -> float:
    """计算单个 mesh 的 Critic plausibility 分数 ∈ [0, 1]（1 = 结构合理）。

    特征采集严格对齐 ``train.py:compute_critic_inputs``：
    1. 渲染多视图 -> CLIP 逐视角编码（augment=False，确定性；推理 CLIPEncoder
       的 training_mode=False，等价于训练侧关闭增强的编码分布）；
    2. 多视角特征平均后重新 L2 归一化（train.aggregate_clip_features）；
    3. ``batch_geometry_features`` 提取 [1, geo_dim] 几何描述子；
    4. ``critic(clip, geo)`` -> sigmoid 概率。

    Args:
        critic: 已加载权重的 SemanticCritic。
        geo_dim: 几何描述子维度（须与训练一致）。
        clip_encoder: CLIPEncoder 实例（Critic 的语义输入来自 CLIP，必需）。
        renderer: 多视角渲染器（image_size 无硬性要求，CLIP 预处理内部 resize）。
        vertices: [1, V, 3] 或 [V, 3]。
        faces: [F, 3] 或 [1, F, 3]。

    Returns:
        float 分数；调用方需保证在 no_grad 上下文中使用。
    """
    verts = vertices.unsqueeze(0) if vertices.dim() == 2 else vertices
    tris = faces[0] if faces.dim() == 3 else faces
    device = verts.device

    out = renderer.render(verts, tris)  # [1, N, 3, H, W]
    images = out["images"]
    batch_size, num_views = images.shape[:2]
    view_features = clip_encoder.encode_images(
        images.flatten(0, 1), augment=False
    ).view(batch_size, num_views, -1)  # [1, N, D]，已 L2 归一化
    clip_features = view_features.mean(dim=1)
    clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    geo_features = batch_geometry_features(verts, tris, geo_dim=geo_dim).to(device)
    score = critic(clip_features.to(device), geo_features)  # [1, 1] sigmoid 概率
    return float(score.flatten()[0].item())


# ====================================================================== #
# 文本引导：候选筛选 + 潜变量优化
# ====================================================================== #
def two_stage_forward(
    generator: MeshGenerator,
    clip: Any,
    renderer: MultiViewRenderer,
    z: Tensor,
    device: torch.device,
    num_feedback_views: int = 2,
) -> Dict[str, Tensor]:
    """带两阶段 CLIP 反馈的生成器前向（与 ``train.two_stage_generate`` 对齐）。

    精修启用（``generator.refinement`` 存在）且 CLIP 可用时：
    1. 第一次 forward（clip_feedback=None）-> 粗 + 细分 mesh；
    2. 细分 mesh 渲染少量视图 -> CLIP 编码（no_grad + detach）-> [B, D] 反馈；
    3. 第二次 forward（clip_feedback=反馈）-> 含精修残差的最终 mesh。

    未启用精修或 CLIP 不可用时退化为单次 forward，旧 checkpoint 行为不变。

    Returns:
        生成器输出 dict；OBJ 导出与渲染使用其中的 ``vertices`` / ``faces``
        （细分 + 精修后的 mesh）。
    """
    if generator.refinement is None or clip is None:
        return generator(z)

    num_views = max(1, min(int(num_feedback_views), renderer.num_views))
    # Stage-1 输出只用于 detached 的反馈信号；梯度只经 Stage-2（同样以 z 为
    # 输入）回传，因此 Stage-1 可安全包在 no_grad 下（optimize_latent 路径亦然）
    with torch.no_grad():
        mesh = generator(z)
    with torch.no_grad():
        camera_poses, _, _ = renderer.generate_camera_poses(z.shape[0], device=device)
        out = renderer.render(
            mesh["vertices"].detach(),
            mesh["faces"].detach(),
            camera_poses=camera_poses[:, :num_views],
        )
        # augment=False：反馈编码绕过随机增强，与训练侧反馈分布一致
        features = clip.encode_images(out["images"].flatten(0, 1), augment=False)
        features = features.view(z.shape[0], num_views, -1).mean(dim=1)
        feedback = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Stage-2 保持有梯度：潜变量优化时梯度经此回传到 latent
    return generator(z, clip_feedback=feedback.detach())


def clip_score_for_mesh(
    clip: Any,
    renderer: MultiViewRenderer,
    text_features: Tensor,
    vertices: Tensor,
    faces: Tensor,
    camera_poses: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """渲染 mesh 并计算与文本的平均 CLIP 相似度。

    Args:
        clip: CLIPEncoder 实例。
        renderer: 多视角渲染器。
        text_features: [1, D] 已归一化的文本特征。
        vertices: [B, V, 3]
        faces: [F, 3] 或 [B, F, 3]
        camera_poses: 可选固定相机位姿 [B, N, 4, 4]。

    Returns:
        (scores [B] 每个样本的平均相似度，images [B, N, 3, H, W] 渲染结果)
        保留计算图，可用于潜变量优化。
    """
    out = renderer.render(vertices, faces, camera_poses=camera_poses)
    images = out["images"]  # [B, N, 3, H, W]
    batch_size, num_views = images.shape[:2]

    image_features = clip.encode_images(images.flatten(0, 1))  # [B*N, D]
    similarity = clip.compute_similarity(image_features, text_features)  # [B*N, 1]
    return similarity.view(batch_size, num_views).mean(dim=1), images


def search_latent(
    generator: MeshGenerator,
    clip: Any,
    renderer: MultiViewRenderer,
    text_features: Tensor,
    num_candidates: int,
    device: torch.device,
    batch_size: int = 4,
) -> Tuple[Tensor, float, List[float]]:
    """采样多个候选 z，按 CLIP 分数挑出与 prompt 最匹配的那个。

    Args:
        num_candidates: 候选数量。
        batch_size: 每次前向的候选数（受显存限制）。

    Returns:
        (best_z [1, latent_dim]，best_score，all_scores 列表)
    """
    best_z: Optional[Tensor] = None
    best_score = -float("inf")
    all_scores: List[float] = []

    with torch.no_grad():
        remaining = max(1, int(num_candidates))
        while remaining > 0:
            current = min(batch_size, remaining)
            remaining -= current

            z = generator.sample_latent(current, device=str(device))
            # 精修启用时按训练路径做两阶段 CLIP 反馈，评分基于最终 mesh
            mesh = two_stage_forward(generator, clip, renderer, z, device)
            scores, _ = clip_score_for_mesh(
                clip, renderer, text_features, mesh["vertices"], mesh["faces"]
            )
            all_scores.extend(float(s) for s in scores)

            top = int(torch.argmax(scores).item())
            if float(scores[top]) > best_score:
                best_score = float(scores[top])
                best_z = z[top : top + 1].clone()

    assert best_z is not None
    return best_z, best_score, all_scores


def optimize_latent(
    generator: MeshGenerator,
    clip: Any,
    renderer: MultiViewRenderer,
    text_features: Tensor,
    z: Tensor,
    steps: int,
    lr: float = 0.02,
    log_every: int = 10,
) -> Tuple[Tensor, float, List[float]]:
    """对潜变量做梯度上升，最大化渲染图与 prompt 的 CLIP 相似度。

    梯度路径：CLIP 图像塔 -> 渲染图像 -> mesh 顶点 -> 生成器 -> z。
    生成器权重保持冻结，只更新 z。

    Returns:
        (优化后的 z [1, latent_dim]，最终分数，每步分数历史)
    """
    latent = z.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)
    history: List[float] = []

    best_z, best_score = latent.detach().clone(), -float("inf")
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        # 反馈在 no_grad 下 detach，梯度仍经 Stage-2 前向回传到 latent
        mesh = two_stage_forward(
            generator, clip, renderer, latent, latent.device
        )
        scores, _ = clip_score_for_mesh(
            clip, renderer, text_features, mesh["vertices"], mesh["faces"]
        )
        loss = (1.0 - scores).mean()  # 相似度越高越好
        loss.backward()
        optimizer.step()

        score = float(scores.mean().item())
        history.append(score)
        if score > best_score:  # 保留历史最优，避免最后一步反而变差
            best_score, best_z = score, latent.detach().clone()
        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"  [优化] step {step + 1}/{steps}  clip_score={score:.4f}")

    return best_z, best_score, history


# ====================================================================== #
# diffusion 后端：种子搜索（替代 GAN 的潜变量搜索）
# ====================================================================== #
def _min_max_normalize(values: List[float]) -> List[float]:
    """把一组分数 min-max 归一到 [0, 1]；全相等时退化为 0.5（combo 用）。"""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.5] * len(values)
    return [(v - low) / (high - low) for v in values]


def search_seed(
    generator: Any,
    renderer: MultiViewRenderer,
    clip_encoder: Any,
    prompt: str,
    config: Dict[str, Any],
    num_candidates: Optional[int] = None,
    critic: Optional[SemanticCritic] = None,
    geo_dim: Optional[int] = None,
    critic_render_cfg: Optional[Dict[str, Any]] = None,
    image: Optional[Path] = None,
) -> Dict[str, Any]:
    """采样 N 个扩散种子生成 mesh，渲染后按配置的 scorer 重排序。

    打分器由 ``inference.scorer`` 控制（默认 ``clip``，保持向后兼容）：

    - ``clip``：渲染图与 prompt 的 CLIP 图文余弦相似度（原行为）；
    - ``critic``：GAN 时代训练的 SemanticCritic plausibility 分数
      （CLIP 多视角聚合特征 + 几何统计描述子，判别结构合理性）；
    - ``combo``：两路分数各自在候选集内 min-max 归一后加权，
      ``score = (1 - w) * clip_norm + w * critic_norm``，
      ``w = inference.critic_weight``。

    critic 打分的预处理严格对齐 train.py：独立构建与 Critic 训练配置一致的
    渲染器（image_size / num_views / camera_distance / elevation_range），
    CLIP 逐视角编码 augment=False 后多视角平均 + 重新 L2 归一化。

    Args:
        generator: ``DiffusionMeshGenerator`` 实例。
        renderer: 多视角渲染器（clip 打分与视图导出用）。
        clip_encoder: CLIPEncoder 实例；critic/combo 打分同样依赖它。
            为 None 且无 critic 时取第一个成功候选（不做重排序）。
        prompt: 文本描述。
        config: diffusion 推理配置；候选数默认读 ``inference.num_candidates``。
        num_candidates: 显式覆盖候选数（None 时用 config 值）。
        critic / geo_dim / critic_render_cfg: ``build_critic`` 的输出；
            scorer 需要 critic 但未提供时自动回退 clip 并告警。
        image: 条件图像路径；提供时透传给 ``generator.generate``，适配器侧
            优先于 prompt 作为 DINOv2 条件（None 时保持纯 prompt 路径不变）。
            注意：重排序打分仍基于 prompt 文本（CLIP 图文相似度），不改。

    Returns:
        最优 mesh dict：``{'vertices', 'faces', 'seed', 'clip_score',
        'critic_score', 'candidate_scores', 'candidate_clip_scores',
        'candidate_critic_scores', 'scorer'}``。

    Raises:
        RuntimeError: 所有候选均生成失败。
    """
    infer_cfg = dict(config.get("inference", {}) or {})
    total = max(1, int(num_candidates or infer_cfg.get("num_candidates", 8)))

    scorer = str(infer_cfg.get("scorer", "clip")).lower()
    if scorer not in ("clip", "critic", "combo"):
        print(f"  [警告] 未知 inference.scorer={scorer!r}，回退 clip")
        scorer = "clip"
    critic_weight = float(infer_cfg.get("critic_weight", 0.5))
    if not 0.0 <= critic_weight <= 1.0:
        print(f"  [警告] inference.critic_weight={critic_weight} 超出 [0, 1]，裁剪到边界")
        critic_weight = min(max(critic_weight, 0.0), 1.0)

    # ---- scorer 可用性检查与回退 ---- #
    need_critic = scorer in ("critic", "combo")
    if need_critic and critic is None:
        print("  [警告] scorer 需要 Critic 但未加载成功，回退 clip 打分")
        scorer = "clip"
    if scorer in ("clip", "combo") and clip_encoder is None:
        if scorer == "combo" and critic is not None:
            print("  [警告] CLIP 不可用，combo 退化为 critic 打分")
            scorer = "critic"
        else:
            print("  [警告] CLIP 不可用且无 Critic，不做重排序，取第一个成功候选")
            scorer = "none"

    # ---- Critic 打分专用渲染器（预处理与 Critic 训练一致）---- #
    critic_renderer: Optional[MultiViewRenderer] = None
    if scorer in ("critic", "combo"):
        critic_renderer = build_renderer(
            {"rendering": dict(critic_render_cfg or {})}, renderer.device
        )
        print(
            f"  [打分器] scorer={scorer} | critic 渲染: "
            f"image_size={critic_renderer.image_size}, num_views={critic_renderer.num_views}"
        )

    text_features: Optional[Tensor] = None
    if clip_encoder is not None:
        text_features = clip_encoder.encode_text([prompt])  # [1, D]，已 L2 归一化

    # 种子池由全局 RNG 派生（generate() 入口已按 --seed 固定），整体可复现
    seeds = torch.randint(0, 2**31 - 1, (total,)).tolist()

    candidates: List[Dict[str, Any]] = []  # {mesh..., seed, clip_score, critic_score}

    for i, cand_seed in enumerate(seeds):
        print(f"  [种子搜索] 候选 {i + 1}/{total}  seed={cand_seed}")
        try:
            # image 为 None 时与旧调用完全等价（适配器 image 缺省即 None）
            mesh = generator.generate(prompt=prompt, seed=int(cand_seed), image=image)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            print(f"  [警告] 候选生成失败（{exc}），跳过")
            continue

        if scorer == "none":
            # 无任何打分器：直接取第一个成功候选
            best = {**mesh, "seed": int(cand_seed), "scorer": "none",
                    "clip_score": None, "critic_score": None,
                    "candidate_scores": [], "candidate_clip_scores": [],
                    "candidate_critic_scores": []}
            print(f"  [种子搜索] 无打分器，直接采用 seed={cand_seed}")
            return best

        entry: Dict[str, Any] = {**mesh, "seed": int(cand_seed),
                                 "clip_score": None, "critic_score": None}

        with torch.no_grad():
            if scorer in ("clip", "combo") and text_features is not None:
                scores, _ = clip_score_for_mesh(
                    clip_encoder, renderer, text_features,
                    mesh["vertices"], mesh["faces"],
                )
                entry["clip_score"] = float(scores.mean().item())

            if scorer in ("critic", "combo") and critic is not None:
                try:
                    entry["critic_score"] = critic_score_for_mesh(
                        critic, int(geo_dim), clip_encoder,
                        critic_renderer, mesh["vertices"], mesh["faces"],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"    [警告] critic 打分失败（{exc}），该候选回退 clip")
        print(
            "    clip_score=%s  critic_score=%s"
            % (
                "None" if entry["clip_score"] is None else f"{entry['clip_score']:.4f}",
                "None" if entry["critic_score"] is None else f"{entry['critic_score']:.4f}",
            )
        )
        candidates.append(entry)

    if not candidates:
        raise RuntimeError("所有种子候选均生成失败，无法完成种子搜索")

    # ---------------- 汇总排序 ---------------- #
    clip_scores = [c["clip_score"] for c in candidates]
    critic_scores = [c["critic_score"] for c in candidates]

    if scorer == "critic":
        # 个别候选 critic 打分失败（None）时以 0 参与排序（最保守）
        rank_scores = [0.0 if s is None else s for s in critic_scores]
    elif scorer == "combo":
        clip_norm = _min_max_normalize(
            [0.0 if s is None else s for s in clip_scores]
        )
        critic_norm = _min_max_normalize(
            [0.0 if s is None else s for s in critic_scores]
        )
        rank_scores = [
            (1.0 - critic_weight) * a + critic_weight * b
            for a, b in zip(clip_norm, critic_norm)
        ]
    else:  # clip
        rank_scores = [-float("inf") if s is None else s for s in clip_scores]

    best_index = max(range(len(candidates)), key=lambda k: rank_scores[k])
    best = candidates[best_index]
    best["scorer"] = scorer
    best["candidate_scores"] = [round(s, 5) for s in rank_scores]
    best["candidate_clip_scores"] = [
        None if s is None else round(s, 5) for s in clip_scores
    ]
    best["candidate_critic_scores"] = [
        None if s is None else round(s, 5) for s in critic_scores
    ]
    print(
        f"  [种子搜索] 最优 seed={best['seed']}  scorer={scorer}  "
        f"rank_score={rank_scores[best_index]:.4f}"
    )
    return best


# ====================================================================== #
# 主流程
# ====================================================================== #
def generate(
    checkpoint_path: Optional[str] = None,
    prompt: str = "",
    output_dir: str = "./results",
    num_views: int = 8,
    image_size: int = 256,
    device: Optional[str] = None,
    num_candidates: int = 8,
    optimize_steps: int = 0,
    optimize_lr: float = 0.02,
    seed: Optional[int] = None,
    grid_cols: Optional[int] = None,
    backend: str = "auto",
    config_path: Optional[str] = None,
    image: Optional[Path] = None,
) -> Dict[str, Any]:
    """完整推理流程：构建生成器 -> 文本引导生成 -> 渲染 -> 导出。

    后端分发：``backend="diffusion"``（或 auto 且 config 中
    ``model.backend == "diffusion"``）走 TripoSG 种子搜索路径；
    否则走 GAN 路径（需 ``checkpoint_path``）。

    Args:
        image: 条件图像路径（仅 diffusion 后端生效）；提供时透传给适配器，
            优先于 prompt 作为 DINOv2 条件。None 时行为与旧版完全一致。

    Returns:
        meta 字典（同时写入 ``*_meta.json``）。
    """
    started = time.time()
    torch_device = resolve_device(device)
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    # 条件图像提前校验（存在性 + 可解码性）：避免进入种子搜索后每个候选
    # 都重复报同样的错误；损坏图片的 PIL.UnidentifiedImageError 在此被
    # 转为 ValueError，由 main 的 except 捕获友好退出
    if image is not None:
        if os.path.isdir(image):
            raise ValueError(f"条件图像应为文件而非目录: {image}")
        if not os.path.isfile(image):
            raise FileNotFoundError(f"条件图像不存在: {image}")
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(image) as _img:
                _img.verify()
        except Exception as exc:
            raise ValueError(f"条件图像无法解码（{exc}）: {image}")

    resolved_backend = resolve_backend(backend, config_path)

    # 两条路径各自填充，汇入下方统一的导出 / 元信息流程
    guided = False
    candidate_scores: List[float] = []
    candidate_clip_scores: List[Optional[float]] = []
    candidate_critic_scores: List[Optional[float]] = []
    optimize_history: List[float] = []
    clip_score: Optional[float] = None
    critic_score_final: Optional[float] = None
    scorer_used: str = "clip"
    meta_source: Dict[str, Any] = {}
    model_config_dump: Dict[str, Any] = {}
    meta_num_candidates = 0
    best_seed: Optional[int] = None

    if resolved_backend == "diffusion":
        # ---------------- diffusion 路径：种子搜索 ---------------- #
        if config_path is None:
            raise ValueError("diffusion 后端需要通过 --config 提供 diffusion 推理配置")
        print(f"[1/3] 加载 diffusion 配置: {config_path}")
        config: Dict[str, Any] = load_yaml_config(config_path)

        print("[2/3] 构建扩散生成器（TripoSG）/ 渲染器 / CLIP")
        generator_diff = build_diffusion_generator(config, torch_device)
        renderer = build_renderer(config, torch_device, num_views, image_size)
        clip = build_clip(config, torch_device)

        guided = clip is not None
        infer_cfg = dict(config.get("inference", {}) or {})
        total_seeds = max(1, int(num_candidates or infer_cfg.get("num_candidates", 8)))

        # ---- Semantic Critic 打分器（可选：inference.scorer = critic/combo）---- #
        scorer = str(infer_cfg.get("scorer", "clip")).lower()
        critic_model: Optional[SemanticCritic] = None
        critic_geo_dim: Optional[int] = None
        critic_render_cfg: Optional[Dict[str, Any]] = None
        critic_checkpoint = infer_cfg.get("critic_checkpoint")
        if scorer in ("critic", "combo"):
            if not critic_checkpoint:
                print("[警告] scorer 需要 inference.critic_checkpoint，未配置，回退 clip")
                scorer = "clip"
            elif clip is None:
                print("[警告] Critic 的语义输入依赖 CLIP，CLIP 不可用，回退无重排序")
                scorer = "clip"
            else:
                try:
                    critic_model, critic_geo_dim, critic_render_cfg = build_critic(
                        str(critic_checkpoint), torch_device
                    )
                    print(
                        f"[2/3] SemanticCritic 已加载: {critic_checkpoint} "
                        f"(clip_dim={critic_model.clip_dim}, geo_dim={critic_model.geo_dim})"
                    )
                except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
                    print(f"[警告] Critic 加载失败（{exc}），回退 clip 打分")
                    scorer = "clip"
        # 回退后同步回 config，保证 search_seed 与 meta 看到的 scorer 一致
        infer_cfg["scorer"] = scorer
        config["inference"] = infer_cfg

        print(f"[3/3] 种子搜索：从 {total_seeds} 个扩散种子中筛选最匹配 prompt 的结果")
        result = search_seed(
            generator_diff, renderer, clip, prompt, config, total_seeds,
            critic=critic_model, geo_dim=critic_geo_dim,
            critic_render_cfg=critic_render_cfg,
            image=image,
        )
        vertices = result["vertices"]  # [1, V, 3]
        faces = result["faces"]  # [F, 3]
        clip_score = result.get("clip_score")
        critic_score_final = result.get("critic_score")
        candidate_scores = list(result.get("candidate_scores", []))
        candidate_clip_scores = list(result.get("candidate_clip_scores", []))
        candidate_critic_scores = list(result.get("candidate_critic_scores", []))
        scorer_used = str(result.get("scorer", scorer))
        best_seed = result.get("seed")
        meta_num_candidates = max(len(candidate_scores), 1 if not guided else total_seeds)
        meta_source = {"config": os.path.abspath(config_path)}
        model_config_dump = dict((config.get("model", {}) or {}).get("diffusion", {}) or {})
    else:
        # ---------------- GAN 路径（原流程，保持不变） ---------------- #
        if image is not None:
            print("[警告] --image 仅 diffusion 后端生效，GAN 后端已忽略该参数")
        if checkpoint_path is None:
            raise ValueError("GAN 后端需要提供 --checkpoint 参数")

        print(f"[1/5] 加载 checkpoint: {checkpoint_path}")
        state = load_checkpoint(checkpoint_path, torch_device)
        config_gan: Dict[str, Any] = state.get("config", {}) or {}

        print("[2/5] 构建生成器 / 渲染器 / CLIP")
        generator = build_generator(config_gan, state["generator"], torch_device)
        renderer = build_renderer(config_gan, torch_device, num_views, image_size)
        clip = build_clip(config_gan, torch_device)

        text_features: Optional[Tensor] = None
        if clip is not None:
            text_features = clip.encode_text([prompt])  # [1, D]，已 L2 归一化

        # ---------------- 文本引导选 z ---------------- #
        guided = clip is not None

        if guided:
            print(f"[3/5] CLIP 引导：从 {num_candidates} 个候选中筛选最匹配 prompt 的潜变量")
            try:
                z, clip_score, candidate_scores = search_latent(
                    generator, clip, renderer, text_features, num_candidates, torch_device
                )
            except RuntimeError as exc:  # 典型原因：没有 nvdiffrast / 非 CUDA 环境
                print(f"[警告] 渲染不可用（{exc}），退化为随机采样，不做文本引导")
                guided = False
                z = generator.sample_latent(1, device=str(torch_device))
            else:
                if optimize_steps > 0:
                    print(f"[3/5] 潜变量优化 {optimize_steps} 步（lr={optimize_lr}）")
                    z, clip_score, optimize_history = optimize_latent(
                        generator,
                        clip,
                        renderer,
                        text_features,
                        z,
                        optimize_steps,
                        optimize_lr,
                    )
        else:
            print("[3/5] 未启用文本引导，直接随机采样潜变量")
            z = generator.sample_latent(1, device=str(torch_device))

        # ---------------- 生成 mesh ---------------- #
        print("[4/5] 生成 mesh 并渲染多视图")
        with torch.no_grad():
            # 精修启用时走两阶段 CLIP 反馈；mesh["vertices"]/["faces"] 即细分后的最终 mesh
            mesh = two_stage_forward(generator, clip, renderer, z, torch_device)
        vertices = mesh["vertices"]  # [1, V, 3]
        faces = mesh["faces"]  # [1, F, 3]
        meta_num_candidates = int(num_candidates) if guided else 0
        candidate_clip_scores = list(candidate_scores)  # GAN 路径候选分即 CLIP 分
        scorer_used = "clip" if guided else "none"
        meta_source = {
            "checkpoint": os.path.abspath(checkpoint_path),
            "checkpoint_iteration": state.get("iteration"),
        }
        model_config_dump = dict(config_gan.get("generator", {}) or {})

    slug = slugify(prompt)
    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, f"{slug}_mesh.obj")
    views_path = os.path.join(output_dir, f"{slug}_views.png")
    meta_path = os.path.join(output_dir, f"{slug}_meta.json")

    export_obj(obj_path, vertices, faces, comment=f"prompt: {prompt}")

    render_error: Optional[str] = None
    try:
        with torch.no_grad():
            out = renderer.render(vertices, faces)
        save_image(tile_images(out["images"][0], ncols=grid_cols), views_path)
    except RuntimeError as exc:  # 渲染失败不影响 mesh 导出
        render_error = str(exc)
        views_path = None
        print(f"[警告] 多视图渲染失败，已跳过图像输出: {exc}")

    # ---------------- 元信息 ---------------- #
    print("[5/5] 写出元信息")
    meta: Dict[str, Any] = {
        "prompt": prompt,
        "prompt_slug": slug,
        "backend": resolved_backend,
    }
    meta.update(meta_source)  # GAN: checkpoint 信息；diffusion: config 路径
    meta.update({
        # json 不能序列化 Path，先转 str；纯 prompt 路径为 None
        "image": os.path.abspath(str(image)) if image else None,
        "device": str(torch_device),
        "seed": best_seed if resolved_backend == "diffusion" else seed,
        "num_views": int(num_views),
        "image_size": int(image_size),
        "num_vertices": int(vertices.shape[1]),
        "num_faces": int(faces.shape[-2]),
        "text_guided": bool(guided),
        "num_candidates": int(meta_num_candidates),
        "scorer": scorer_used,
        "candidate_clip_scores": candidate_clip_scores,
        "candidate_critic_scores": candidate_critic_scores,
        "candidate_scores": [round(s, 5) for s in candidate_scores],
        "optimize_steps": int(optimize_steps) if (guided and resolved_backend == "gan") else 0,
        "optimize_lr": float(optimize_lr),
        "optimize_history": [round(s, 5) for s in optimize_history],
        "clip_score": None if clip_score is None else round(float(clip_score), 5),
        "critic_score": None if critic_score_final is None else round(float(critic_score_final), 5),
        "generator_config": model_config_dump,
        "elapsed_seconds": round(time.time() - started, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "outputs": {
            "mesh": os.path.abspath(obj_path),
            "views": os.path.abspath(views_path) if views_path else None,
        },
    })
    if render_error:
        meta["render_error"] = render_error

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print(f"完成: mesh={obj_path}")
    if views_path:
        print(f"      views={views_path}")
    print(f"      meta={meta_path}")
    if clip_score is not None:
        print(f"      clip_score={clip_score:.4f}")
    return meta


# ====================================================================== #
# 入口
# ====================================================================== #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic3D-GAN 推理脚本：文本 -> 3D mesh + 多视图渲染"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="训练好的 checkpoint 路径（GAN 后端必需）"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "gan", "diffusion"],
        help="推理后端；auto 时从 --config 的 model.backend 探测，缺省 gan",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="diffusion 推理配置路径（如 configs/diffusion_inference.yaml）",
    )
    parser.add_argument("--prompt", type=str, required=True, help="文本描述，如 'a red chair'")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="条件图像路径；提供时优先于 prompt 作为 DINOv2 条件（diffusion 后端生效）",
    )
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")
    parser.add_argument("--num_views", type=int, default=8, help="渲染视角数量")
    parser.add_argument("--image_size", type=int, default=256, help="单视图渲染分辨率")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动选择")
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=8,
        help="CLIP 引导的候选潜变量数量（生成器无条件，靠打分筛选实现文本控制）",
    )
    parser.add_argument(
        "--optimize_steps", type=int, default=0, help="潜变量 CLIP 梯度优化步数，0 表示关闭"
    )
    parser.add_argument("--optimize_lr", type=float, default=0.02, help="潜变量优化学习率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--grid_cols", type=int, default=None, help="多视图拼图列数")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        generate(
            checkpoint_path=args.checkpoint,
            prompt=args.prompt,
            output_dir=args.output_dir,
            num_views=args.num_views,
            image_size=args.image_size,
            device=args.device,
            num_candidates=args.num_candidates,
            optimize_steps=args.optimize_steps,
            optimize_lr=args.optimize_lr,
            seed=args.seed,
            grid_cols=args.grid_cols,
            backend=args.backend,
            config_path=args.config,
            image=Path(args.image) if args.image else None,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

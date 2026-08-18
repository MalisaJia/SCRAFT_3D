"""CRaFT-3D 推理脚本：文本 -> 3D mesh（diffusion 后端）。

后端为 TripoSG rectified-flow 扩散模型（见 models/diffusion_adapter.py）。
TripoSG 本体无文本条件，文本控制通过**种子搜索**实现：采样 N 个扩散噪声
种子分别生成 mesh，渲染多视图后重排序，取最高分结果（``search_seed``）。
打分器由 ``inference.scorer`` 选择：``clip``（CLIP 图文相似度，默认）/
``critic``（SemanticCritic 结构合理性分数）/ ``combo``（两者 min-max
归一后加权）/ ``combo_prod``（两者 min-max 归一后相乘，双门槛）。

用法：
    python inference.py --config configs/diffusion_inference.yaml --prompt "a red chair" \
        --output_dir ./results --num_views 8 --image_size 256

输出（``{output_dir}/{prompt_slug}_*``）：
    *_mesh.obj   OBJ 格式 mesh（顶点 + 面）
    *_views.png  多视图拼图
    *_meta.json  生成参数与打分记录
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# 允许从任意工作目录运行本脚本
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.semantic_critic import SemanticCritic  # noqa: E402
from rendering.multi_view_render import MultiViewRenderer  # noqa: E402
from utils.geometry_features import batch_geometry_features  # noqa: E402
from utils.visualize import save_image, tile_images  # noqa: E402

try:  # 适配器导入失败时给出明确错误提示，而不是在模块导入期直接崩
    # （TripoSG 本体在 DiffusionMeshGenerator 实例化时才懒加载导入）
    from models.diffusion_adapter import DiffusionMeshGenerator

    _DIFFUSION_IMPORTABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    DiffusionMeshGenerator = None  # type: ignore[assignment]
    _DIFFUSION_IMPORTABLE = False

try:  # yaml 用于读取 diffusion 推理 config
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
    lines: List[str] = ["# CRaFT-3D generated mesh"]
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
    """读取 YAML 配置文件为 dict。

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


def resolve_backend(
    backend: Optional[str] = None,
    config_path: Optional[str] = None,
) -> str:
    """确定推理后端。

    CRaFT-3D 只有 diffusion 一条生成路径，本函数保留为兼容性入口，
    参数均被忽略，恒返回 ``"diffusion"``。
    """
    return "diffusion"


# ====================================================================== #
# 模型构建
# ====================================================================== #
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
    """从 checkpoint 加载 SemanticCritic（种子重排的结构合理性打分器）。

    checkpoint 中 critic 权重位于 ``state["critic"]``；``clip_dim / geo_dim /
    hidden_dim / dropout`` 优先取 checkpoint 内保存的 ``config.semantic_critic``，
    保证结构与训练完全一致。注意：构建时必须先建模型再 load_state_dict——
    spectral_norm 的 ``weight_u / weight_v`` 缓冲只有在构造时才会生成。

    Args:
        checkpoint_path: 含 critic 权重的 checkpoint 路径。
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

    特征采集与 Critic 训练时的预处理一致：
    1. 渲染多视图 -> CLIP 逐视角编码（augment=False，确定性）；
    2. 多视角特征平均后重新 L2 归一化；
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
# CLIP 打分
# ====================================================================== #
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
    """
    out = renderer.render(vertices, faces, camera_poses=camera_poses)
    images = out["images"]  # [B, N, 3, H, W]
    batch_size, num_views = images.shape[:2]

    image_features = clip.encode_images(images.flatten(0, 1))  # [B*N, D]
    similarity = clip.compute_similarity(image_features, text_features)  # [B*N, 1]
    return similarity.view(batch_size, num_views).mean(dim=1), images


# ====================================================================== #
# 种子搜索：扩散噪声种子重排序
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
) -> Dict[str, Any]:
    """采样 N 个扩散种子生成 mesh，渲染后按配置的 scorer 重排序。

    打分器由 ``inference.scorer`` 控制（默认 ``clip``，保持向后兼容）：

    - ``clip``：渲染图与 prompt 的 CLIP 图文余弦相似度（原行为）；
    - ``critic``：SemanticCritic plausibility 分数
      （CLIP 多视角聚合特征 + 几何统计描述子，判别结构合理性）；
    - ``combo``：两路分数各自在候选集内 min-max 归一后加权，
      ``score = (1 - w) * clip_norm + w * critic_norm``，
      ``w = inference.critic_weight``。
    - ``combo_prod``：两路分数各自在候选集内 min-max 归一后相乘（双门槛：
      语义与结构缺一不可），``score = clip_norm * critic_norm + 1e-9``。

    critic 打分的预处理与 Critic 训练侧对齐：独立构建与 Critic 训练配置一致的
    渲染器（image_size / num_views / camera_distance / elevation_range），
    CLIP 逐视角编码 augment=False 后多视角平均 + 重新 L2 归一化。

    Args:
        generator: ``DiffusionMeshGenerator`` 实例。
        renderer: 多视角渲染器（clip 打分与视图导出用）。
        clip_encoder: CLIPEncoder 实例；critic/combo/combo_prod 打分同样依赖它
            （critic 的语义输入就是 CLIP 多视图特征）。为 None 时所有打分器
            都不可用，不做重排序，取第一个成功候选。
        prompt: 文本描述。
        config: diffusion 推理配置；候选数默认读 ``inference.num_candidates``。
        num_candidates: 显式覆盖候选数（None 时用 config 值）。
        critic / geo_dim / critic_render_cfg: ``build_critic`` 的输出；
            scorer 需要 critic 但未提供时自动回退 clip 并告警。

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
    if scorer not in ("clip", "critic", "combo", "combo_prod"):
        print(f"  [警告] 未知 inference.scorer={scorer!r}，回退 clip")
        scorer = "clip"
    critic_weight = float(infer_cfg.get("critic_weight", 0.5))
    if not 0.0 <= critic_weight <= 1.0:
        print(f"  [警告] inference.critic_weight={critic_weight} 超出 [0, 1]，裁剪到边界")
        critic_weight = min(max(critic_weight, 0.0), 1.0)

    # ---- scorer 可用性检查与回退 ---- #
    need_critic = scorer in ("critic", "combo", "combo_prod")
    if need_critic and critic is None:
        print("  [警告] scorer 需要 Critic 但未加载成功，回退 clip 打分")
        scorer = "clip"
    if clip_encoder is None:
        # critic_score_for_mesh 本身依赖 CLIP 多视图特征：CLIP 缺失时
        # critic/combo/combo_prod 全部无法打分（"退化为 critic" 不成立），
        # 如实落到 none，绝不让打分异常被吞掉后静默乱序
        if scorer in ("critic", "combo", "combo_prod"):
            print(
                f"  [警告] CLIP 不可用；critic 打分也依赖 CLIP 特征，"
                f"{scorer} 无法重排序，取第一个成功候选"
            )
        else:
            print("  [警告] CLIP 不可用，不做重排序，取第一个成功候选")
        scorer = "none"

    # ---- Critic 打分专用渲染器（预处理与 Critic 训练一致）---- #
    critic_renderer: Optional[MultiViewRenderer] = None
    if scorer in ("critic", "combo", "combo_prod"):
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
            mesh = generator.generate(prompt=prompt, seed=int(cand_seed))
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
            if scorer in ("clip", "combo", "combo_prod") and text_features is not None:
                scores, _ = clip_score_for_mesh(
                    clip_encoder, renderer, text_features,
                    mesh["vertices"], mesh["faces"],
                )
                entry["clip_score"] = float(scores.mean().item())

            if scorer in ("critic", "combo", "combo_prod") and critic is not None:
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
    elif scorer == "combo_prod":
        # 双门槛：两路各自 min-max 归一后相乘（max==min 时归一退化为全 0.5）
        clip_norm = _min_max_normalize(
            [0.0 if s is None else s for s in clip_scores]
        )
        critic_norm = _min_max_normalize(
            [0.0 if s is None else s for s in critic_scores]
        )
        rank_scores = [a * b + 1e-9 for a, b in zip(clip_norm, critic_norm)]
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
    prompt: str = "",
    output_dir: str = "./results",
    num_views: int = 8,
    image_size: int = 256,
    device: Optional[str] = None,
    num_candidates: int = 8,
    seed: Optional[int] = None,
    grid_cols: Optional[int] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """完整推理流程：构建扩散生成器 -> 种子搜索 -> 渲染 -> 导出。

    Returns:
        meta 字典（同时写入 ``*_meta.json``）。
    """
    started = time.time()
    torch_device = resolve_device(device)
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    if config_path is None:
        raise ValueError("需要通过 --config 提供 diffusion 推理配置")
    print(f"[1/5] 加载 diffusion 配置: {config_path}")
    config: Dict[str, Any] = load_yaml_config(config_path)

    print("[2/5] 构建扩散生成器（TripoSG）/ 渲染器 / CLIP")
    generator_diff = build_diffusion_generator(config, torch_device)
    renderer = build_renderer(config, torch_device, num_views, image_size)
    clip = build_clip(config, torch_device)

    guided = clip is not None
    infer_cfg = dict(config.get("inference", {}) or {})
    total_seeds = max(1, int(num_candidates or infer_cfg.get("num_candidates", 8)))

    # ---- Semantic Critic 打分器（可选：inference.scorer = critic/combo/combo_prod）---- #
    scorer = str(infer_cfg.get("scorer", "clip")).lower()
    critic_model: Optional[SemanticCritic] = None
    critic_geo_dim: Optional[int] = None
    critic_render_cfg: Optional[Dict[str, Any]] = None
    critic_checkpoint = infer_cfg.get("critic_checkpoint")
    if scorer in ("critic", "combo", "combo_prod"):
        if not critic_checkpoint:
            print("[警告] scorer 需要 inference.critic_checkpoint，未配置，回退 clip")
            scorer = "clip"
        elif clip is None:
            # 与 search_seed 的回退链一致：critic 依赖 CLIP 特征，
            # CLIP 缺失时没有任何可用打分器 -> 无重排序（而非回退 clip）
            print("[警告] Critic 的语义输入依赖 CLIP，CLIP 不可用，回退无重排序")
            scorer = "none"
        else:
            try:
                critic_model, critic_geo_dim, critic_render_cfg = build_critic(
                    str(critic_checkpoint), torch_device
                )
                print(
                    f"[2/5] SemanticCritic 已加载: {critic_checkpoint} "
                    f"(clip_dim={critic_model.clip_dim}, geo_dim={critic_model.geo_dim})"
                )
            except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
                print(f"[警告] Critic 加载失败（{exc}），回退 clip 打分")
                scorer = "clip"
    # 回退后同步回 config，保证 search_seed 与 meta 看到的 scorer 一致
    infer_cfg["scorer"] = scorer
    config["inference"] = infer_cfg

    print(f"[3/5] 种子搜索：从 {total_seeds} 个扩散种子中筛选最匹配 prompt 的结果")
    result = search_seed(
        generator_diff, renderer, clip, prompt, config, total_seeds,
        critic=critic_model, geo_dim=critic_geo_dim,
        critic_render_cfg=critic_render_cfg,
    )
    vertices = result["vertices"]  # [1, V, 3]
    faces = result["faces"]  # [F, 3]
    clip_score: Optional[float] = result.get("clip_score")
    critic_score_final: Optional[float] = result.get("critic_score")
    candidate_scores: List[float] = list(result.get("candidate_scores", []))
    candidate_clip_scores: List[Optional[float]] = list(
        result.get("candidate_clip_scores", [])
    )
    candidate_critic_scores: List[Optional[float]] = list(
        result.get("candidate_critic_scores", [])
    )
    scorer_used = str(result.get("scorer", scorer))
    best_seed: Optional[int] = result.get("seed")
    meta_num_candidates = max(len(candidate_scores), 1 if not guided else total_seeds)
    model_config_dump = dict((config.get("model", {}) or {}).get("diffusion", {}) or {})

    print("[4/5] 导出 mesh 与多视图")
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
        "backend": "diffusion",
        "config": os.path.abspath(config_path),
        "device": str(torch_device),
        "seed": best_seed,
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
        "clip_score": None if clip_score is None else round(float(clip_score), 5),
        "critic_score": None if critic_score_final is None else round(float(critic_score_final), 5),
        "generator_config": model_config_dump,
        "elapsed_seconds": round(time.time() - started, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "outputs": {
            "mesh": os.path.abspath(obj_path),
            "views": os.path.abspath(views_path) if views_path else None,
        },
    }
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
        description="CRaFT-3D 推理脚本：文本 -> 3D mesh + 多视图渲染"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="diffusion 推理配置路径（如 configs/diffusion_inference.yaml）",
    )
    parser.add_argument("--prompt", type=str, required=True, help="文本描述，如 'a red chair'")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")
    parser.add_argument("--num_views", type=int, default=8, help="渲染视角数量")
    parser.add_argument("--image_size", type=int, default=256, help="单视图渲染分辨率")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动选择")
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=8,
        help="种子搜索的候选数量（扩散模型无文本条件，靠打分重排序实现文本控制）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--grid_cols", type=int, default=None, help="多视图拼图列数")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        generate(
            prompt=args.prompt,
            output_dir=args.output_dir,
            num_views=args.num_views,
            image_size=args.image_size,
            device=args.device,
            num_candidates=args.num_candidates,
            seed=args.seed,
            grid_cols=args.grid_cols,
            config_path=args.config,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

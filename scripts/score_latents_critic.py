"""Latent 缓存的 Critic 打分脚本（Learned Semantic Reward 阶段二）。

读取 ``scripts/precompute_latents.py`` 输出的 latent shard（含 latents/uids），
逐样本 ``decode_latent_to_mesh`` -> 多视图渲染 -> critic_v2 打分，把分数
软裁剪、均值归一，并可选线性锐化放大（--sharpen）：

    weight = clip(raw_score, 0.5, 1.5) / mean(全部已打分样本)
    weight = clip(1 + sharpen * (weight - 1), clip_min, clip_max)
    weight = weight / mean(weight)   # 乘性再归一，均值精确为 1.0

``--sharpen 1.0`` 时第二步为恒等变换，行为与旧版完全一致。

可选 ``--semantic_prompts``（逗号分隔提示文本）启用 critic 门槛 × CLIP 语义
双奖励：语义分（复用打分渲染视图，视图×提示余弦相似度均值）在全体样本内
min-max 归一后转为均值 1 的乘性调制 m∈约[0.5,1.5]，final = w_q * m 归一后
clip 到 [weight_clip_min, weight_clip_max] 再乘性归一；语义分续跑状态落盘
``clip_sem_scores.json``。不提供该参数时全流程与旧版逐条一致。

输出两个文件：

- ``cache_dir/critic_scores.json``：内部续跑状态，``{uid: raw_score}``；
  已打分 uid 自动跳过，中断后可直接重跑续接。
- ``cache_dir/critic_weights.json``：最终训练权重 ``{uid: weight}``
  （整批均值归一到 1.0），供 ``train_diffusion.py`` 的
  ``training.critic_weights_path`` 使用。

用法（服务器上执行）::

    python scripts/score_latents_critic.py \
        --cache_dir /root/autodl-tmp/3D-gans/cache/triposg_latents \
        --critic outputs/critic_v2.pt [--limit 256]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# 允许从任意工作目录运行本脚本
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference import build_clip, build_critic, critic_score_for_mesh  # noqa: E402
from rendering.multi_view_render import MultiViewRenderer  # noqa: E402
from train_diffusion import (  # noqa: E402
    _resolve_weights_dir,
    decode_latent_to_mesh,
    load_config,
)
from utils.geometry_features import batch_geometry_features  # noqa: E402

# 权重软裁剪区间（论文约定：低分样本降权但不完全剔除）
WEIGHT_MIN = 0.5
WEIGHT_MAX = 1.5

# Critic 打分专用八叉树深度：保持打分口径与既有 critic_weights.json 一致，
# 不随 train_diffusion 的训练内评估预览深度（EVAL_OCTREE_DEPTH）变化
SCORING_OCTREE_DEPTH = 7

SCORES_FILENAME = "critic_scores.json"
WEIGHTS_FILENAME = "critic_weights.json"
# CLIP 语义调节分数的续跑状态文件（仅 --semantic_prompts 提供时落盘）
SEM_SCORES_FILENAME = "clip_sem_scores.json"


# ====================================================================== #
# VAE 加载（只取 VAE，不加载 DiT 主干，省显存）
# ====================================================================== #
def load_vae(config: Dict[str, Any], device: torch.device) -> Any:
    """按训练配置加载冻结的 TripoSG VAE（与 build_dit_with_lora 同一权重源）。"""
    try:
        from triposg.models.autoencoders import TripoSGVAEModel
    except ImportError as exc:
        raise RuntimeError(
            "未能导入 TripoSG（triposg.models.autoencoders）。"
            "请先克隆 TripoSG 仓库并安装其依赖，或检查 PYTHONPATH。"
        ) from exc

    diffusion_cfg = (config.get("model", {}) or {}).get("diffusion", {})
    weights_path = str(diffusion_cfg.get("weights_path", "") or "")
    vae = None
    weights_dir = _resolve_weights_dir(weights_path)
    if weights_dir is not None and (weights_dir / "vae" / "config.json").is_file():
        vae = TripoSGVAEModel.from_pretrained(str(weights_dir), subfolder="vae")
    if vae is None:
        from triposg.pipelines import TripoSGPipeline

        print("本地 VAE 权重不可用，回退 HuggingFace VAST-AI/TripoSG ...")
        vae = TripoSGPipeline.from_pretrained("VAST-AI/TripoSG").vae
    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


# ====================================================================== #
# 缓存扫描
# ====================================================================== #
def scan_cache(cache_dir: str) -> List[Tuple[str, str, int]]:
    """展开 manifest -> [(shard 文件名, uid, shard 内下标)]，顺序与预计算一致。"""
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"找不到 {manifest_path}，请先运行 scripts/precompute_latents.py"
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    entries: List[Tuple[str, str, int]] = []
    for shard_info in manifest.get("shards", []):
        shard_name = shard_info["path"]
        shard_path = os.path.join(cache_dir, shard_name)
        if not os.path.isfile(shard_path):
            print(f"[警告] manifest 中的 shard 不存在，跳过: {shard_path}")
            continue
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        uids = list(shard.get("uids", []))
        num = int(shard_info.get("num_samples", shard["latents"].shape[0]))
        for i in range(num):
            uid = str(uids[i]) if i < len(uids) else f"{shard_name}:{i}"
            entries.append((shard_name, uid, i))
    return entries


def scores_to_weights(
    scores: Dict[str, float],
    sharpen: float = 1.0,
    clip_min: float = WEIGHT_MIN,
    clip_max: float = WEIGHT_MAX,
) -> Dict[str, float]:
    """raw 分数 -> 训练权重：软裁剪 -> 均值归一 -> 线性锐化放大 -> 再归一。

    ``sharpen=1.0`` 时锐化步骤为恒等变换（放大后权重仍在 [0.5, 1.5] 内，
    二次裁剪与再归一都不改变结果），与旧版行为完全一致。
    """
    if not scores:
        return {}
    clipped = {uid: min(max(float(value), WEIGHT_MIN), WEIGHT_MAX) for uid, value in scores.items()}
    mean = sum(clipped.values()) / len(clipped)
    if mean <= 0.0:
        return {uid: 1.0 for uid in clipped}
    weights = {uid: value / mean for uid, value in clipped.items()}

    sharpen = float(sharpen)
    if sharpen != 1.0:
        # 线性放大偏离 1.0 的部分，再裁剪回安全区间，避免极端权重
        weights = {
            uid: min(max(1.0 + sharpen * (value - 1.0), clip_min), clip_max)
            for uid, value in weights.items()
        }
        # 乘性再归一：均值精确回到 1.0（裁剪后均值可能略偏）
        sharp_mean = sum(weights.values()) / len(weights)
        if sharp_mean > 0.0:
            weights = {uid: value / sharp_mean for uid, value in weights.items()}
    return weights


def weight_stats(weights: Dict[str, float]) -> str:
    """权重统计行：条数 / mean / min / max / std（便于确认信号强度）。"""
    values = [float(v) for v in weights.values()]
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return (
        f"count={n} mean={mean:.4f} min={min(values):.4f} "
        f"max={max(values):.4f} std={std:.4f}"
    )


def semantic_modulation(
    sem_scores: Dict[str, float], uids: List[str]
) -> Dict[str, float]:
    """CLIP 语义分数 -> 均值 1 的乘性调制因子 m（纯函数，便于单测）。

    全体样本内 min-max 归一得 n∈[0,1]（max==min 时 n=0.5 防除零），
    m0 = 0.5 + n ∈ [0.5, 1.5]，再 m = m0 / mean(m0) 使均值精确 1.0。
    只对 ``uids`` 中且存在于 ``sem_scores`` 的样本计算。
    """
    vals = [float(sem_scores[uid]) for uid in uids if uid in sem_scores]
    if not vals:
        return {}
    low, high = min(vals), max(vals)
    span = high - low
    m0: Dict[str, float] = {}
    for uid in uids:
        if uid not in sem_scores:
            continue
        n = 0.5 if span < 1e-12 else (float(sem_scores[uid]) - low) / span
        m0[uid] = 0.5 + n
    m_mean = sum(m0.values()) / len(m0)
    return {uid: v / m_mean for uid, v in m0.items()}


def combine_dual_weights(
    critic_scores: Dict[str, float],
    sem_scores: Dict[str, float],
    sharpen: float = 1.0,
    clip_min: float = WEIGHT_MIN,
    clip_max: float = WEIGHT_MAX,
) -> Tuple[Dict[str, float], int]:
    """critic 门槛 × CLIP 语义调节的双奖励组合（纯函数）。

    1. ``w_q = scores_to_weights(critic_scores, sharpen, ...)``（均值 1）；
    2. 语义调制 ``m``（见 :func:`semantic_modulation`，均值 1）；
    3. ``final = w_q * m`` 归一到均值 1 -> clip 到 [clip_min, clip_max]
       -> 再乘性归一，均值精确 1.0。

    只覆盖两路分数都齐全的 uid。
    Returns:
        ``({uid: final_weight}, saturated)``。饱和计数在 clip 之后、
        再归一之前统计（对再归一后的权重比对边界会系统性漏检裁剪样本）。
    """
    w_q = scores_to_weights(critic_scores, sharpen=sharpen,
                            clip_min=clip_min, clip_max=clip_max)
    uids = [uid for uid in w_q if uid in sem_scores]
    if not uids:
        return {}, 0
    m = semantic_modulation(sem_scores, uids)
    final_raw = {uid: w_q[uid] * m[uid] for uid in uids}
    raw_mean = sum(final_raw.values()) / len(final_raw)
    final = {uid: v / raw_mean for uid, v in final_raw.items()}
    final = {uid: min(max(v, clip_min), clip_max) for uid, v in final.items()}
    # 饱和 = 裁剪后、再归一前贴 clip 边界的样本数（诊断口径与裁剪动作对齐）
    tol = 1e-8
    saturated = sum(
        1 for v in final.values()
        if v <= clip_min + tol or v >= clip_max - tol
    )
    clip_mean = sum(final.values()) / len(final)
    if clip_mean > 0.0:
        final = {uid: v / clip_mean for uid, v in final.items()}
    return final, saturated


def flush_state(
    scores: Dict[str, float],
    scores_path: str,
    out_path: str,
    sharpen: float = 1.0,
    clip_min: float = WEIGHT_MIN,
    clip_max: float = WEIGHT_MAX,
) -> None:
    """续跑状态 + 训练权重一起增量落盘（崩溃也不丢已打分结果）。"""
    with open(scores_path, "w", encoding="utf-8") as handle:
        json.dump(scores, handle, ensure_ascii=False)
    weights = scores_to_weights(scores, sharpen=sharpen, clip_min=clip_min, clip_max=clip_max)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(weights, handle, ensure_ascii=False)


def load_sem_state(
    sem_scores_path: str, semantic_prompts: List[str]
) -> Dict[str, float]:
    """读取 CLIP 语义分续跑状态（新格式 ``{"prompts": [...], "scores": {...}}``）。

    - 新格式且 prompts 与本次 ``--semantic_prompts`` 不一致 -> RuntimeError
      （不同提示词算出的语义分不可混用，须删除旧状态重跑）；
    - 旧版扁平格式 ``{uid: float}``（无 prompts 元数据可校验）-> 丢弃重算并警告。
    """
    with open(sem_scores_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if (
        isinstance(data, dict)
        and isinstance(data.get("prompts"), list)
        and isinstance(data.get("scores"), dict)
    ):
        old_prompts = [str(p) for p in data["prompts"]]
        if old_prompts != [str(p) for p in semantic_prompts]:
            raise RuntimeError(
                f"{sem_scores_path} 记录的 semantic_prompts 与本次不一致："
                f"旧={old_prompts}，新={list(semantic_prompts)}。"
                f"请删除该文件（及 critic_weights.json）后重跑"
            )
        return {str(k): float(v) for k, v in data["scores"].items()}
    print(
        f"[警告] {sem_scores_path} 为旧版扁平格式（无 prompts 元数据，无法校验"
        f"提示词一致性），语义分将丢弃并重新计算"
    )
    return {}


def flush_dual_state(
    scores: Dict[str, float],
    sem_scores: Dict[str, float],
    cache_dir: str,
    out_path: str,
    sharpen: float,
    clip_min: float,
    clip_max: float,
    prompts: Optional[List[str]] = None,
) -> None:
    """双奖励模式的增量落盘：critic / CLIP 语义两路续跑状态 + 组合权重。

    ``critic_weights.json`` 的文件格式与单奖励模式完全一致（训练侧零改动）；
    组合权重只覆盖两路分数都齐全的 uid。语义分续跑状态写新格式
    ``{"prompts": [...], "scores": {...}}``。防线：critic 分数非空而组合
    权重为空时抛 RuntimeError——绝不写空权重文件覆盖有效权重。
    """
    weights, _ = combine_dual_weights(
        scores, sem_scores, sharpen=sharpen, clip_min=clip_min, clip_max=clip_max
    )
    if scores and not weights:
        raise RuntimeError(
            "双奖励组合权重为空（critic 分数非空但没有任何 uid 两路分数齐全），"
            "拒绝写空权重文件，请检查语义打分是否全部失败"
        )
    with open(os.path.join(cache_dir, SCORES_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(scores, handle, ensure_ascii=False)
    sem_state = {"prompts": list(prompts or []), "scores": sem_scores}
    with open(os.path.join(cache_dir, SEM_SCORES_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(sem_state, handle, ensure_ascii=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(weights, handle, ensure_ascii=False)


@torch.no_grad()
def score_mesh_dual(
    vae: Any,
    latent: Tensor,
    octree_depth: int,
    critic: Any,
    geo_dim: int,
    clip_encoder: Any,
    renderer: Any,
    device: torch.device,
    prompt_text_features: Optional[Tensor] = None,
) -> Tuple[float, Optional[float]]:
    """单样本打分：解码 -> 渲染一次 -> critic 分（必有）+ CLIP 语义分（可选）。

    语义分复用 critic 打分已渲染的多视图（确定性渲染策略不变）：逐视角 CLIP
    特征（augment=False）与每个 semantic prompt 的文本特征算余弦相似度，
    对 视图×提示 取均值。``prompt_text_features`` 为 None 时语义分返回 None。

    Returns:
        (critic_score, sem_score_or_None)
    """
    vertices, faces = decode_latent_to_mesh(vae, latent, octree_depth)
    vertices = vertices.to(device)
    faces = faces.to(device)

    out = renderer.render(vertices.unsqueeze(0), faces)
    images = out["images"]  # [1, N, 3, H, W]
    batch_size, num_views = images.shape[:2]
    view_features = clip_encoder.encode_images(
        images.flatten(0, 1), augment=False
    ).view(batch_size, num_views, -1)  # [1, N, D]，已 L2 归一化

    clip_features = view_features.mean(dim=1)
    clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    geo_features = batch_geometry_features(
        vertices.unsqueeze(0), faces, geo_dim=geo_dim
    ).to(device)
    critic_score = float(critic(clip_features.to(device), geo_features).flatten()[0].item())

    sem_score: Optional[float] = None
    if prompt_text_features is not None:
        # 与 inference.clip_score_for_mesh 同款相似度接口：[N 视图, P 提示] 取均值
        sims = clip_encoder.compute_similarity(
            view_features.reshape(-1, view_features.shape[-1]), prompt_text_features
        )
        sem_score = float(sims.mean().item())
    return critic_score, sem_score


# ====================================================================== #
# 主流程
# ====================================================================== #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 critic_v2 给 latent 缓存打分并产出训练权重"
        "（Learned Semantic Reward 阶段二）"
    )
    parser.add_argument("--cache_dir", type=str, required=True, help="latent shard 缓存目录")
    parser.add_argument(
        "--critic",
        type=str,
        default="outputs/critic_v2.pt",
        help="Critic checkpoint（finetune_critic.py 输出的 critic_v2.pt）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_diffusion.yaml",
        help="训练配置（读取 model.diffusion.weights_path 定位 TripoSG VAE）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="权重输出路径（默认 cache_dir/critic_weights.json）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="本次运行最多新打分多少个样本（0 = 全部；配合续跑先验证子集）",
    )
    parser.add_argument(
        "--octree_depth",
        type=int,
        default=SCORING_OCTREE_DEPTH,
        help="打分用八叉树深度（默认与既有 critic_weights.json 口径一致）",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动")
    parser.add_argument("--flush_every", type=int, default=50, help="续跑状态+权重增量落盘间隔")
    parser.add_argument(
        "--sharpen",
        type=float,
        default=3.0,
        help="权重信号锐化放大系数：w' = 1 + sharpen * (w - 1)。"
        "1.0 = 与旧版完全一致；默认 3.0 用于拉开 critic 打分聚拢的动态范围",
    )
    parser.add_argument(
        "--weight_clip_min",
        type=float,
        default=WEIGHT_MIN,
        help=f"锐化后权重的下限裁剪（默认 {WEIGHT_MIN}）",
    )
    parser.add_argument(
        "--weight_clip_max",
        type=float,
        default=WEIGHT_MAX,
        help=f"锐化后权重的上限裁剪（默认 {WEIGHT_MAX}）",
    )
    parser.add_argument(
        "--semantic_prompts",
        type=str,
        default=None,
        help="CLIP 语义调节的提示文本，逗号分隔（如 'a chair,a wooden chair'）。"
        "提供后启用 critic 门槛 × CLIP 语义双奖励：语义分对全体样本 min-max "
        "归一后作为乘性调制（续跑状态落盘到 clip_sem_scores.json）；"
        "不提供时全流程与旧行为逐条一致",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not (math.isfinite(args.weight_clip_min) and args.weight_clip_min > 0.0):
        print(
            f"[错误] --weight_clip_min ({args.weight_clip_min}) 必须为大于 0 的有限数，"
            f"否则会产生零/负权重，破坏均值=1.0 不变量",
            file=sys.stderr,
        )
        return 1
    if args.weight_clip_min >= args.weight_clip_max:
        print(
            f"[错误] --weight_clip_min ({args.weight_clip_min}) 必须小于 "
            f"--weight_clip_max ({args.weight_clip_max})",
            file=sys.stderr,
        )
        return 1
    sharpen = float(args.sharpen)
    if not (math.isfinite(sharpen) and sharpen > 0.0):
        print(
            f"[错误] --sharpen 必须为正有限数（nan/inf 不被接受），当前: {sharpen}",
            file=sys.stderr,
        )
        return 1
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"设备: {device}")
    print(
        f"权重锐化: sharpen={sharpen}"
        + ("" if sharpen == 1.0 else f"（信号放大 {sharpen:.1f} 倍）")
        + f"，锐化后裁剪区间 [{args.weight_clip_min}, {args.weight_clip_max}]"
    )

    out_path = args.out or os.path.join(args.cache_dir, WEIGHTS_FILENAME)
    scores_path = os.path.join(args.cache_dir, SCORES_FILENAME)

    # ---- Critic + 与训练预处理一致的打分渲染器 ---- #
    critic, geo_dim, render_cfg = build_critic(args.critic, device)
    critic_state = torch.load(args.critic, map_location="cpu", weights_only=False)
    critic_config: Dict[str, Any] = dict(critic_state.get("config", {}) or {})
    elevation = tuple(render_cfg.get("elevation_range", (-30.0, 30.0)))
    renderer = MultiViewRenderer(
        image_size=int(render_cfg.get("image_size", 256)),
        num_views=int(render_cfg.get("num_views", 4)),
        camera_distance=float(render_cfg.get("camera_distance", 2.5)),
        elevation_range=(float(elevation[0]), float(elevation[1])),
        # fixed：打分要求确定性（与 inference.py 的 critic 打分路径一致）
        azimuth_strategy="fixed",
        device=str(device),
    )
    clip_encoder = build_clip(critic_config, device)
    if clip_encoder is None:
        print("[错误] CLIP 不可用，无法运行 Critic 打分", file=sys.stderr)
        return 1

    # ---- CLIP 语义调节（可选：--semantic_prompts；None 时旧行为逐条不变）---- #
    semantic_prompts: Optional[List[str]] = None
    if args.semantic_prompts is not None:
        semantic_prompts = [p.strip() for p in args.semantic_prompts.split(",")
                            if p.strip()]
        if not semantic_prompts:
            print("[错误] --semantic_prompts 解析后为空，请检查逗号分隔格式",
                  file=sys.stderr)
            return 1
    sem_enabled = semantic_prompts is not None
    sem_text_features = (
        clip_encoder.encode_text(list(semantic_prompts)) if sem_enabled else None
    )  # [P, D]，已 L2 归一化；prompts 只编码一次，全样本复用
    if sem_enabled:
        print(
            f"权重双奖励: CLIP 语义调节启用，semantic_prompts={semantic_prompts}"
            f"（复用打分渲染视图，{len(semantic_prompts)} 条提示）"
        )

    # ---- VAE + latent 缓存 ---- #
    config = load_config(args.config)
    vae = load_vae(config, device)
    entries = scan_cache(args.cache_dir)
    print(f"latent 缓存共 {len(entries)} 个样本: {args.cache_dir}")

    # ---- 续跑状态：已打分 uid 直接跳过 ---- #
    scores: Dict[str, float] = {}
    scores_file_exists = os.path.isfile(scores_path)
    if scores_file_exists:
        with open(scores_path, "r", encoding="utf-8") as handle:
            scores = {str(k): float(v) for k, v in json.load(handle).items()}
        print(f"已有 {len(scores)} 个已打分样本，自动续跑")

    sem_scores: Dict[str, float] = {}
    sem_scores_path = os.path.join(args.cache_dir, SEM_SCORES_FILENAME)
    sem_file_exists = os.path.isfile(sem_scores_path)
    if sem_enabled and sem_file_exists:
        # 新格式校验 prompts 一致性；旧版扁平格式丢弃重算（见 load_sem_state）
        sem_scores = load_sem_state(sem_scores_path, semantic_prompts)
        if sem_scores:
            print(f"已有 {len(sem_scores)} 个 CLIP 语义分样本，自动续跑")

    # 续跑对账：两路状态文件都存在时打印 critic / 语义差集统计
    if sem_enabled and scores_file_exists and sem_file_exists:
        missing_sem = sum(1 for uid in scores if uid not in sem_scores)
        if missing_sem > 0:
            print(
                f"[续跑对账] critic 分 {len(scores)} 条 / 语义分 {len(sem_scores)} 条："
                f"{missing_sem} 个 uid 有 critic 分但缺语义分，将重新打分补齐（覆写幂等）"
            )
        else:
            print(f"[续跑对账] critic 分 {len(scores)} 条与语义分状态一致，无缺失")

    shard_cache: Dict[str, Dict[str, Any]] = {}
    newly_scored = 0
    failed = 0
    limit = int(args.limit)

    for shard_name, uid, inner_idx in entries:
        # 双奖励模式下必须两路分数都齐全才跳过：只有 critic 分而缺语义分
        # （旧跑未开 --semantic_prompts / 两次落盘之间中断）的 uid 会重新
        # 打分补齐，scores[uid] 覆写幂等
        if uid in scores and (not sem_enabled or uid in sem_scores):
            continue
        if limit > 0 and newly_scored >= limit:
            print(f"已达到 --limit {limit}，本次先到这里（重跑即可继续）")
            break

        if shard_name not in shard_cache:
            shard_cache[shard_name] = torch.load(
                os.path.join(args.cache_dir, shard_name),
                map_location="cpu",
                weights_only=False,
            )
        latent = shard_cache[shard_name]["latents"][inner_idx].float().to(device)

        try:
            if sem_enabled:
                # 双奖励：解码 -> 渲染一次 -> critic 分 + CLIP 语义分（视图复用）
                score, sem_score = score_mesh_dual(
                    vae, latent, args.octree_depth, critic, geo_dim,
                    clip_encoder, renderer, device,
                    prompt_text_features=sem_text_features,
                )
                sem_scores[uid] = float(sem_score)
            else:
                vertices, faces = decode_latent_to_mesh(vae, latent, args.octree_depth)
                # decode_latent_to_mesh 返回 CPU 张量；渲染器 / CLIP / Critic 都在
                # CUDA 上，必须先搬到 device，否则 nvdiffrast 报 non-CUDA DeviceType
                score = critic_score_for_mesh(
                    critic, geo_dim, clip_encoder, renderer,
                    vertices.to(device), faces.to(device)
                )
        except Exception as exc:  # 单样本任何异常都跳过，绝不让整个打分 pass 崩掉
            failed += 1
            print(f"[警告] uid={uid} 解码/打分失败，跳过: {exc}")
            traceback.print_exc()
            continue

        scores[uid] = float(score)
        newly_scored += 1
        if newly_scored % max(args.flush_every, 1) == 0:
            if sem_enabled:
                flush_dual_state(
                    scores, sem_scores, args.cache_dir, out_path,
                    sharpen=sharpen,
                    clip_min=args.weight_clip_min,
                    clip_max=args.weight_clip_max,
                    prompts=semantic_prompts,
                )
            else:
                flush_state(
                    scores, scores_path, out_path,
                    sharpen=sharpen,
                    clip_min=args.weight_clip_min,
                    clip_max=args.weight_clip_max,
                )
            print(f"  进度: 本次新打分 {newly_scored}（累计 {len(scores)}，失败 {failed}）")

    # ---- 落盘：续跑状态 + 最终训练权重 ---- #
    saturated = 0
    if sem_enabled:
        # 防线在 flush_dual_state 内：critic 分非空而组合权重为空时抛
        # RuntimeError 退出，绝不用空权重覆盖有效文件
        flush_dual_state(
            scores, sem_scores, args.cache_dir, out_path,
            sharpen=sharpen,
            clip_min=args.weight_clip_min,
            clip_max=args.weight_clip_max,
            prompts=semantic_prompts,
        )
        weights, saturated = combine_dual_weights(
            scores, sem_scores,
            sharpen=sharpen,
            clip_min=args.weight_clip_min,
            clip_max=args.weight_clip_max,
        )
    else:
        flush_state(
            scores, scores_path, out_path,
            sharpen=sharpen,
            clip_min=args.weight_clip_min,
            clip_max=args.weight_clip_max,
        )
        weights = scores_to_weights(
            scores,
            sharpen=sharpen,
            clip_min=args.weight_clip_min,
            clip_max=args.weight_clip_max,
        )

    if weights:
        print(
            f"完成: 打分 {len(weights)} 个样本（本次新增 {newly_scored}，失败 {failed}）"
        )
        if sem_enabled:
            # 双奖励诊断：w_q / m / final 三段统计 + final 贴边界饱和率
            dual_uids = [uid for uid in scores if uid in sem_scores]
            w_q = scores_to_weights(
                scores, sharpen=sharpen,
                clip_min=args.weight_clip_min, clip_max=args.weight_clip_max,
            )
            m = semantic_modulation(sem_scores, dual_uids)
            print(f"[双奖励 w_q] {weight_stats({u: w_q[u] for u in dual_uids})}")
            print(f"[双奖励 m ] {weight_stats(m)}")
            print(f"[双奖励 final] {weight_stats(weights)}")
            # saturated 来自 combine_dual_weights：clip 之后、再归一之前统计，
            # 与裁剪动作口径一致（再归一后比对边界会系统性漏检）
            sat_rate = saturated / len(weights)
            print(f"[双奖励 final 饱和率] {saturated}/{len(weights)} = "
                  f"{sat_rate * 100:.2f}%（贴 weight_clip 边界）")
            if sat_rate > 0.20:
                print(
                    f"[WARNING] final 权重饱和率 {sat_rate * 100:.1f}% > 20%，"
                    f"双奖励组合过激，建议降低 --sharpen 或收窄 "
                    f"[--weight_clip_min, --weight_clip_max]"
                )
        else:
            print(f"[最终权重统计] {weight_stats(weights)}")
    else:
        print("[警告] 没有任何成功打分的样本，未产出权重")
    print(f"续跑状态: {scores_path}")
    print(f"训练权重: {out_path}（train_diffusion.py 的 training.critic_weights_path 指向它）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

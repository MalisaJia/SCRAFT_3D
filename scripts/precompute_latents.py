"""Precompute TripoSG VAE latents from Objaverse dataset.

Usage:
    python scripts/precompute_latents.py --config configs/train_diffusion.yaml [--num_samples 2000]

Output: saves latent tensors + captions as .pt shards in config.data.latent_cache_dir

设计要点：
- **懒加载 TripoSG**：与 ``models/diffusion_adapter.py`` 相同的模式 —— TripoSG
  相关模块仅在真正需要时才导入，未安装的环境下本脚本之外的仓库代码不受影响。
- **断点续跑**：扫描输出目录已有 shard 中记录的 uid，跳过已处理的样本。
- **shard 分片**：每 ``--shard_size`` 个样本落盘一次 ``latent_shard_{i}.pt``，
  包含 ``{'latents', 'captions', 'uids'}`` 三个键（若成功预计算了 DINOv2
  图像条件嵌入则额外含 ``'image_embeds'``）；结束时（及每片落盘后）刷新
  ``manifest.json`` 元数据（样本总数、latent 形状、各 shard 信息）。
- **mesh 归一化**：与渲染器 / 适配器一致 —— 包围盒中心化 + 缩放到单位球内。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor

# 允许从任意工作目录运行本脚本（把项目根目录加入 import 路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.objaverse import ObjaverseDataset  # noqa: E402
from datasets.shapenet import load_obj  # noqa: E402

# tqdm 为可选依赖：缺失时退化为定期打印进度
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:  # pragma: no cover - 取决于运行环境
    HAS_TQDM = False

# 每个 mesh 采样的表面点数（VAE encode 的输入规模）
DEFAULT_NUM_SURFACE_POINTS = 8192

# latent 以 fp16 落盘：显存占用减半，对 VAE 解码精度的影响可忽略
LATENT_DTYPE = torch.float16


# ====================================================================== #
# 配置与命令行
# ====================================================================== #
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Precompute TripoSG VAE latents from Objaverse meshes"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_diffusion.yaml",
        help="扩散训练配置文件（读取 data / model.diffusion 段）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="最多处理的样本数（默认处理标注文件中的全部样本）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="样本数上限（与 --num_samples 取较小值；达到即停并正常写 manifest）",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="mesh 相对路径的解析根目录（默认为 annotations.json 所在目录）",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=256,
        help="每个 shard 包含的样本数（默认 256）",
    )
    parser.add_argument(
        "--num_surface_points",
        type=int,
        default=DEFAULT_NUM_SURFACE_POINTS,
        help="每个 mesh 采样的表面点数（默认 8192）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="VAE 编码设备（默认 cuda）",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """加载 YAML 配置（缺失的顶层 block 补成空 dict）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for block in ("model", "training", "data", "logging"):
        config.setdefault(block, {})
    return config


# ====================================================================== #
# TripoSG VAE 懒加载（与 diffusion_adapter.py 同一模式）
# ====================================================================== #
@torch.no_grad()
def _fps_fallback(points: torch.Tensor, keep: int) -> torch.Tensor:
    """纯 PyTorch 贪心最远点采样（torch_cluster.fps 的替身）。

    Args:
        points: [M, 3] 候选点坐标。
        keep: 保留点数。

    Returns:
        被选中的点下标 [K]（long，CPU）。
    """
    device = points.device
    num_points = points.shape[0]
    keep = min(keep, num_points)
    selected = torch.zeros(keep, dtype=torch.long)
    indices = torch.arange(num_points, device=device)
    current = points[0]
    min_dist = torch.full((num_points,), float("inf"), device=device)
    for i in range(keep):
        selected[i] = int(indices[0].item())
        min_dist = torch.minimum(min_dist, ((points - current) ** 2).sum(dim=-1))
        order = torch.argsort(min_dist, descending=True)
        points = points[order]
        indices = indices[order]
        min_dist = min_dist[order]
        current = points[0]
    return selected


def _patch_vae_fps(vae: Any) -> None:
    """为上游 VAE 的 ``_sample_features`` 打 FPS 兼容补丁。

    官方 ``autoencoder_kl_triposg.py`` 中 ``from torch_cluster import fps`` 被
    注释掉但代码仍调用 ``fps``，未安装 torch_cluster 时 encode 会招
    ``NameError``。这里用贪心最远点采样的纯 PyTorch 实现替换
    ``_sample_features``（先随机取 4x 候选再 FPS 降采到 num_tokens，
    与上游语义一致）。
    """
    if getattr(vae, "_fps_patched", False):
        return

    def _sample_features(
        x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1]
        )
        selected = x[:, indices]  # [B, 4K, C]
        batch_size, _, num_channels = selected.shape
        out = []
        for b in range(batch_size):
            keep = _fps_fallback(selected[b, :, :3], num_tokens)
            out.append(selected[b, keep.to(selected.device)])
        return torch.stack(out, dim=0).view(batch_size, -1, num_channels)

    vae._sample_features = _sample_features  # type: ignore[method-assign]
    vae._fps_patched = True  # type: ignore[attr-defined]
    print("[信息] 已为 TripoSG VAE 启用纯 PyTorch FPS 回退（torch_cluster 未安装）")


def load_triposg_vae(weights_path: str, device: torch.device):
    """懒加载 TripoSG VAE（``TripoSGVAEModel``，diffusers 目录布局）并冻结。

    Args:
        weights_path: diffusers 布局权重目录（含 ``model_index.json`` 与
            ``vae/`` 子目录）；留空则从 HuggingFace ``VAST-AI/TripoSG`` 拉取。
        device: VAE 所在设备。

    Returns:
        冻结后的 ``TripoSGVAEModel`` 实例（eval 模式）。

    Raises:
        RuntimeError: TripoSG 未安装。
    """
    try:
        from triposg.models.autoencoders import TripoSGVAEModel
    except ImportError as exc:
        raise RuntimeError(
            "未能导入 TripoSG（triposg.models.autoencoders.TripoSGVAEModel）。"
            "请先克隆 TripoSG 仓库并安装其依赖，或检查 PYTHONPATH 是否包含 "
            "triposg 目录。"
        ) from exc

    if weights_path:
        weights_dir = Path(weights_path)
        if weights_dir.is_file():
            weights_dir = weights_dir.parent
        if not (weights_dir / "model_index.json").is_file():
            raise FileNotFoundError(
                f"权重目录缺少 model_index.json（diffusers 目录布局）: {weights_dir}"
            )
        vae = TripoSGVAEModel.from_pretrained(str(weights_dir), subfolder="vae")
    else:
        # 未提供本地权重时尝试 HuggingFace 官方仓库
        from triposg.pipelines import TripoSGPipeline

        pipeline = TripoSGPipeline.from_pretrained("VAST-AI/TripoSG")
        vae = pipeline.vae

    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    _patch_vae_fps(vae)
    return vae


# DINOv2 图像条件的官方口径 token 数：feature_extractor_dinov2 把图 resize +
# center-crop 到 224×224，patch_size=14 -> (224/14)^2 = 256 patches + 1 CLS = 257。
# 推理管线 encode_image 运行时实测 [1, 257, 1024]。预计算必须与之一致。
EXPECTED_DINO_NUM_TOKENS = 257


def verify_dinov2_calibration(processor, encoder) -> int:
    """校验 DINOv2 processor 口径，返回 dummy 图过编码器后的 token 数。

    用一张 dummy 512×512 图走一遍 processor + encoder，统计 last_hidden_state
    的 token 数；必须等于官方 TripoSG feature_extractor 的 257（224/center-crop，
    patch14 -> 256 patches + 1 CLS）。不等则直接 RuntimeError 退出，防止再次
    出现训练/推理口径漂移（历史上曾因手工覆盖 518px 产出 1370 tokens 导致
    LoRA 生成团块）。
    """
    from PIL import Image

    size_cfg = getattr(processor, "size", None)
    crop_cfg = getattr(processor, "crop_size", None)
    dummy = Image.new("RGB", (512, 512), color=(127, 127, 127))
    inputs = processor(images=dummy, return_tensors="pt")
    device = next(encoder.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        tokens = int(encoder(**inputs).last_hidden_state.shape[1])
    print(
        f"[口径校验] DINOv2 processor size={size_cfg} crop_size={crop_cfg} "
        f"| dummy 512x512 -> {tokens} tokens（官方口径应为 {EXPECTED_DINO_NUM_TOKENS}）",
        flush=True,
    )
    if tokens != EXPECTED_DINO_NUM_TOKENS:
        raise RuntimeError(
            f"DINOv2 口径漂移：dummy 512x512 得到 {tokens} tokens，"
            f"预期官方 {EXPECTED_DINO_NUM_TOKENS}（224/center-crop/patch14）。"
            "请确认 feature_extractor_dinov2 与推理管线 encode_image 同源，"
            "且未对 processor 做任何 size/crop 覆盖。"
        )
    return tokens


def load_dinov2_encoder(weights_path: str, device: torch.device):
    """懒加载 pipeline 自带的 DINOv2 图像编码器 + 官方特征提取器（可选）。

    用于把每个 mesh 的参考图（若存在）编码为 cross-attention 条件嵌入；
    权重目录缺少 image_encoder_dinov2 子目录时返回 None（训练侧退化到
    零嵌入无条件分支）。

    processor 一律用 ``AutoImageProcessor.from_pretrained`` 直接加载权重目录
    下的官方 ``feature_extractor_dinov2`` 子目录（224/center-crop，257 tokens），
    与推理侧 ``models/diffusion_adapter.py`` / 官方 pipeline 的 encode_image 同源；
    **禁止**对 size/crop_size 做任何手工覆盖。加载成功后立即用
    ``verify_dinov2_calibration`` 以 dummy 图校验 token 数，漂移则 RuntimeError。
    """
    weights_dir = Path(weights_path) if weights_path else None
    if weights_dir is not None and weights_dir.is_file():
        weights_dir = weights_dir.parent
    if weights_dir is None or not (weights_dir / "image_encoder_dinov2").is_dir():
        print("[信息] 权重目录缺少 image_encoder_dinov2，跳过图像嵌入预计算")
        return None

    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            str(weights_dir / "feature_extractor_dinov2")
        )
        encoder = AutoModel.from_pretrained(str(weights_dir / "image_encoder_dinov2"))
        encoder = encoder.to(device).eval()
        for param in encoder.parameters():
            param.requires_grad_(False)
    except Exception as exc:
        print(f"[警告] DINOv2 编码器加载失败（{exc}），跳过图像嵌入预计算")
        return None

    # 口径校验放在 try/except 之外：漂移必须 RuntimeError 直接退出，不能被兜底吞掉
    verify_dinov2_calibration(processor, encoder)
    return processor, encoder


# ====================================================================== #
# Mesh 归一化与表面采样
# ====================================================================== #
def canonicalize_mesh(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """包围盒中心化 + 缩放到单位球内（与 diffusion_adapter 的归一化一致）。"""
    vertices = np.asarray(vertices, dtype=np.float32)
    center = 0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
    vertices = vertices - center
    radius = float(np.linalg.norm(vertices, axis=1).max())
    if radius < 1e-8:
        raise RuntimeError("mesh 退化：所有顶点重合，无法归一化")
    return vertices / radius, faces


def sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    normal_noise: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """按面积加权从 mesh 表面采样点与法线。

    Args:
        vertices: [V, 3] 归一化后的顶点。
        faces: [F, 3] 面索引。
        num_points: 采样点数。
        normal_noise: 法线扰动幅度（避免大量共面点法线完全相同）。

    Returns:
        (points [N, 3], normals [N, 3])，均为 float32。
    """
    tri = vertices[faces]  # [F, 3, 3]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)  # [F, 3]，模长 = 2 * 三角形面积
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = areas.sum()
    if total < 1e-12:
        raise RuntimeError("mesh 退化：总面积为零")
    probs = areas / total

    face_idx = np.random.choice(len(faces), size=num_points, replace=True, p=probs)
    # 三角形内均匀采样（sqrt 技巧）
    r1 = np.sqrt(np.random.rand(num_points).astype(np.float32))
    r2 = np.random.rand(num_points).astype(np.float32)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2

    sampled_tri = tri[face_idx]  # [N, 3, 3]
    points = (
        w0[:, None] * sampled_tri[:, 0]
        + w1[:, None] * sampled_tri[:, 1]
        + w2[:, None] * sampled_tri[:, 2]
    )

    normals = cross[face_idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    normals = normals + np.random.randn(*normals.shape).astype(np.float32) * normal_noise
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points.astype(np.float32), normals.astype(np.float32)


# ====================================================================== #
# VAE 编码（真实签名：encode(x=[B, N, 6] cat([xyz, normals]), num_tokens)）
# ====================================================================== #
def encode_mesh_latent(
    vae: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    device: torch.device,
    num_surface_points: int,
    num_tokens: int = 2048,
) -> Tensor:
    """把一个 mesh 编码为 VAE latent。

    真实 ``TripoSGVAEModel.encode`` 输入为表面点云（坐标与法线拼接成
    [B, N, 6]），返回 ``AutoencoderKLOutput(latent_dist=...)``；对其
    ``.sample()`` 得到 [B, 2048, 64] 的形状 latent。

    Args:
        vae: 冻结的 TripoSGVAEModel。
        vertices: [V, 3] 已归一化顶点。
        faces: [F, 3] 面索引。
        device: 编码设备。
        num_surface_points: 表面采样点数。
        num_tokens: latent token 数（与预训练 / 训练配置一致，默认 2048）。

    Returns:
        latent 张量（去掉 batch 维），形状 [2048, 64]。
    """
    points, normals = sample_surface_points(vertices, faces, num_surface_points)
    points_t = torch.from_numpy(points).unsqueeze(0).to(device)  # [1, N, 3]
    normals_t = torch.from_numpy(normals).unsqueeze(0).to(device)  # [1, N, 3]
    surface = torch.cat([points_t, normals_t], dim=-1)  # [1, N, 6]

    ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), ctx:
        output = vae.encode(surface, num_tokens=num_tokens)
        latent = output.latent_dist.sample()

    if latent.dim() == 3 and latent.shape[0] == 1:
        latent = latent.squeeze(0)  # [1, 2048, 64] -> [2048, 64]
    return latent.detach().float().to(LATENT_DTYPE).cpu()


# ====================================================================== #
# Shard 读写与断点续跑
# ====================================================================== #
def scan_existing_shards(cache_dir: str) -> Tuple[Set[str], int]:
    """扫描已有 shard，返回 (已处理 uid 集合, 下一个 shard 编号)。"""
    processed: Set[str] = set()
    max_index = -1
    if not os.path.isdir(cache_dir):
        return processed, 0

    for name in sorted(os.listdir(cache_dir)):
        if not (name.startswith("latent_shard_") and name.endswith(".pt")):
            continue
        try:
            shard = torch.load(os.path.join(cache_dir, name), map_location="cpu")
        except Exception as exc:  # 损坏的 shard 直接忽略，重跑时会重新编号覆盖
            print(f"[警告] 读取 shard {name} 失败（{exc}），跳过")
            continue
        try:
            index = int(name[len("latent_shard_") : -len(".pt")])
        except ValueError:
            index = max_index + 1
        max_index = max(max_index, index)
        processed.update(str(uid) for uid in shard.get("uids", []))
    return processed, max_index + 1


def flush_shard(
    buffer: Dict[str, Any], cache_dir: str, shard_index: int, manifest: Dict[str, Any]
) -> str:
    """把累积的样本写入 ``latent_shard_{i}.pt`` 并刷新 manifest。"""
    os.makedirs(cache_dir, exist_ok=True)
    shard_path = os.path.join(cache_dir, f"latent_shard_{shard_index}.pt")
    shard = {
        "latents": torch.stack(buffer["latents"]),  # [N, 2048, 64] fp16
        "captions": buffer["captions"],  # List[str]
        "uids": buffer["uids"],  # List[str]
    }
    # DINOv2 图像条件嵌入（可选）：仅当本片全部样本都有嵌入时才落盘
    embeds = buffer.get("image_embeds", [])
    if embeds and len(embeds) == len(buffer["uids"]):
        shard["image_embeds"] = torch.stack(embeds)
    torch.save(shard, shard_path)

    manifest["shards"].append(
        {
            "path": os.path.basename(shard_path),
            "num_samples": len(buffer["uids"]),
            "uids": list(buffer["uids"]),
        }
    )
    manifest["num_samples"] += len(buffer["uids"])
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    buffer["latents"].clear()
    buffer["captions"].clear()
    buffer["uids"].clear()
    if "image_embeds" in buffer:
        buffer["image_embeds"].clear()
    return shard_path


# ====================================================================== #
# 参考图 -> DINOv2 嵌入（可选，与 mesh 同名同目录的图片视为参考图）
# ====================================================================== #
_REFERENCE_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def find_reference_image(mesh_path: str) -> Optional[str]:
    """查找与 mesh 同名（不同扩展名）的参考图，不存在返回 None。"""
    stem = Path(mesh_path).with_suffix("")
    for ext in _REFERENCE_IMAGE_EXTS:
        candidate = Path(str(stem) + ext)
        if candidate.is_file():
            return str(candidate)
    return None


@torch.no_grad()
def encode_image_embeds(
    dinov2: Any, image_path: str, device: torch.device
) -> Optional[Tensor]:
    """用 pipeline 同款 DINOv2 编码器把参考图编码为条件嵌入 [S, 1024]。"""
    from PIL import Image

    processor, encoder = dinov2
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with ctx:
        outputs = encoder(**inputs)
    embeds = outputs.last_hidden_state.squeeze(0).float()  # [S, 1024]
    return embeds.to(LATENT_DTYPE).cpu()


# ====================================================================== #
# 主流程
# ====================================================================== #
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    diffusion_cfg = (config.get("model", {}) or {}).get("diffusion", {})

    cache_dir = str(data_cfg.get("latent_cache_dir", "./cache/triposg_latents"))
    annotations_path = str(data_cfg.get("annotations_path", "./data/objaverse/annotations.json"))

    # ---- 加载 Objaverse 标注（复用 datasets/objaverse.py 的解析逻辑）----
    data_root = args.data_root or os.path.dirname(os.path.abspath(annotations_path))
    dataset = ObjaverseDataset(data_root=data_root, annotation_file=annotations_path)
    entries = list(dataset.entries)
    if not entries:
        raise RuntimeError(
            f"标注文件 {annotations_path} 中没有任何样本，请检查路径与格式 "
            "（期望 {uid: {path, category, caption, lvis}}）"
        )
    if args.num_samples is not None:
        entries = entries[: args.num_samples]
    if args.max_samples is not None:
        entries = entries[: args.max_samples]
    print(f"[数据] 标注样本总数: {len(entries)}（data_root={data_root}）")

    # ---- 断点续跑：跳过已处理 uid ----
    processed_uids, next_shard_index = scan_existing_shards(cache_dir)
    pending = [e for e in entries if e["uid"] not in processed_uids]
    print(
        f"[续跑] 已处理 {len(processed_uids)} 个样本，"
        f"剩余待处理 {len(pending)} 个（shard 编号从 {next_shard_index} 开始）"
    )
    if not pending:
        print("[完成] 所有样本均已处理，无需重新编码")
        return

    # ---- 加载冻结 VAE 与可选的 DINOv2 图像编码器 ----
    device = torch.device(args.device)
    weights_path = str(diffusion_cfg.get("weights_path", "") or "")
    print(f"[模型] 加载 TripoSG VAE（weights_path={weights_path or 'HuggingFace'}）...")
    vae = load_triposg_vae(weights_path, device)
    dinov2 = load_dinov2_encoder(weights_path, device)

    # ---- 编码循环 ----
    # 断点续跑：先加载已有 manifest，避免首次 flush 时覆盖历史 shard 引用
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest: Dict[str, Any] = json.load(fh)
        # 补齐缺失的元数据键（shards 列表必须保留）
        manifest.setdefault("num_samples", 0)
        manifest.setdefault("shards", [])
        manifest.setdefault("latent_shape", None)
        manifest.setdefault("latent_dtype", str(LATENT_DTYPE).split(".")[-1])
        manifest.setdefault("num_surface_points", args.num_surface_points)
    else:
        manifest = {
            "num_samples": 0,
            "latent_shape": None,
            "latent_dtype": str(LATENT_DTYPE).split(".")[-1],
            "num_surface_points": args.num_surface_points,
            "shards": [],
        }
    buffer: Dict[str, Any] = {"latents": [], "captions": [], "uids": [], "image_embeds": []}
    shard_index = next_shard_index
    num_encoded = 0
    num_failed = 0

    iterator = tqdm(pending, desc="编码 latent", unit="mesh") if HAS_TQDM else pending
    for entry in iterator:
        mesh_path = dataset._resolve_path(entry["path"])
        try:
            if not os.path.isfile(mesh_path):
                raise FileNotFoundError(f"mesh 文件不存在: {mesh_path}")
            vertices, faces = load_obj(mesh_path)
            if vertices.size == 0 or faces.size == 0:
                raise RuntimeError("空 mesh")
            vertices, faces = canonicalize_mesh(vertices, faces)
            latent = encode_mesh_latent(
                vae, vertices, faces, device, args.num_surface_points
            )
        except Exception as exc:
            num_failed += 1
            if not HAS_TQDM or num_failed <= 20:
                print(f"[警告] uid={entry['uid']} 编码失败（{exc}），跳过")
            continue

        if manifest["latent_shape"] is None:
            manifest["latent_shape"] = list(latent.shape)
        elif list(latent.shape) != manifest["latent_shape"]:
            print(
                f"[警告] uid={entry['uid']} latent 形状 {list(latent.shape)} "
                f"与首个样本 {manifest['latent_shape']} 不一致，跳过"
            )
            num_failed += 1
            continue

        buffer["latents"].append(latent)
        buffer["captions"].append(entry["caption"])
        buffer["uids"].append(entry["uid"])
        num_encoded += 1

        # 可选：参考图 -> DINOv2 条件嵌入（任一失败则本片不带嵌入落盘）
        if dinov2 is not None:
            ref_image = find_reference_image(mesh_path)
            if ref_image is not None:
                try:
                    embeds = encode_image_embeds(dinov2, ref_image, device)
                    buffer["image_embeds"].append(embeds)
                except Exception as exc:
                    print(f"[警告] uid={entry['uid']} 图像嵌入编码失败（{exc}）")
                    buffer["image_embeds"].clear()

        # shard 满了就落盘（manifest 同步刷新，中断也不丢进度）
        if len(buffer["uids"]) >= args.shard_size:
            shard_path = flush_shard(buffer, cache_dir, shard_index, manifest)
            print(f"[落盘] {shard_path}（累计 {manifest['num_samples']} 样本）")
            shard_index += 1

    # 尾片（不足 shard_size 的剩余样本）
    if buffer["uids"]:
        shard_path = flush_shard(buffer, cache_dir, shard_index, manifest)
        print(f"[落盘] {shard_path}（尾片，累计 {manifest['num_samples']} 样本）")

    print(
        f"[完成] 新编码 {num_encoded} 个样本，失败 {num_failed} 个；"
        f"缓存目录共 {manifest['num_samples']} 个样本 -> {cache_dir}"
    )


if __name__ == "__main__":
    main()

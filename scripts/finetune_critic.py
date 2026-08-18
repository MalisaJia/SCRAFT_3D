"""Critic 继续教育脚本（Learned Semantic Reward 阶段一）。

把 GAN 时代训练好的 SemanticCritic 在新的正负样本池上继续微调，产出
``critic_v2.pt``：

- **正样本**：``--mesh_dir`` 下的真实 mesh（obj/glb 混合）归一化到单位球后，
  按旧 Critic checkpoint 内保存的训练渲染配置（默认 256 / 4 视图 / cam 2.5 /
  el ±30）渲染多视图 -> 冻结 CLIP 逐视图编码 -> 视图平均 + re-L2-norm，
  叠加几何统计描述子，label = 1。可选 ``--generate_triposg N`` 用本地
  TripoSG 现采 N 个 mesh 补充正样本（本地跑不了 TripoSG 时自动跳过，
  只依赖 ``--mesh_dir``）。
- **负样本**：复用旧 ``AnomalyConstructor`` 四策略，外加两种新损坏算子
  （顶点高斯噪声、随机 decimate / 面片翻转），label = 0；正负比例
  ``--pos_ratio`` 可配（默认 1:1）。
- **只训 Critic MLP**：CLIP 全程冻结（特征离线预计算），BCE loss，
  默认 3000 步；按源 mesh 留出 10% 验证集，结束时打印 train/val accuracy。
- **保存格式与旧 checkpoint 完全一致**：``{'critic': state_dict,
  'config': {...}}``，``inference.py:build_critic`` 可原样加载。

用法（服务器上执行）::

    python scripts/finetune_critic.py \
        --mesh_dir /root/autodl-tmp/objaverse_data/meshes \
        --old_critic outputs/ablation_a0_full/ckpt_final.pt \
        --out outputs/critic_v2.pt --steps 3000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# 允许从任意工作目录运行本脚本
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference import build_critic  # noqa: E402
from models.semantic_critic import SemanticCritic  # noqa: E402
from rendering.multi_view_render import MultiViewRenderer  # noqa: E402
from utils.anomaly_constructor import AnomalyConstructor, _bbox_diagonal  # noqa: E402
from utils.geometry_features import batch_geometry_features  # noqa: E402
from vlm.clip_encoder import CLIPEncoder  # noqa: E402

MESH_EXTENSIONS = (".obj", ".glb", ".gltf", ".ply", ".stl")

# 新增损坏算子名称（与 AnomalyConstructor 四策略共同组成负样本策略池）
NEW_CORRUPTION_STRATEGIES = ("vertex_noise", "face_flip", "decimate")


# ====================================================================== #
# 网格读取 / 归一化
# ====================================================================== #
def normalize_to_unit_sphere(vertices: np.ndarray) -> np.ndarray:
    """bbox 中心对齐 + 最大半径缩放到单位球（与 decode_latent_to_mesh 一致）。"""
    center = 0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
    vertices = vertices - center
    radius = float(np.linalg.norm(vertices, axis=1).max())
    if radius < 1e-8:
        raise ValueError("mesh 退化（半径为 0）")
    return vertices / radius


def load_mesh_file(path: str) -> Optional[Tuple[Tensor, Tensor]]:
    """读取单个 mesh 文件并归一化；失败返回 None（跳过而不是中断）。"""
    try:
        import trimesh

        scene = trimesh.load(path, force="mesh", process=True)
        vertices = np.asarray(scene.vertices, dtype=np.float32)
        faces = np.asarray(scene.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            print(f"[警告] 空 mesh，跳过: {path}")
            return None
        vertices = normalize_to_unit_sphere(vertices)
        return (
            torch.from_numpy(np.ascontiguousarray(vertices)),
            torch.from_numpy(np.ascontiguousarray(faces)),
        )
    except Exception as exc:  # 文件格式损坏 / 缺解码器
        print(f"[警告] 读取失败，跳过 {path}: {exc}")
        return None


def scan_mesh_dir(mesh_dir: str) -> List[str]:
    """递归扫描目录下的 mesh 文件（obj/glb 混合）。"""
    paths: List[str] = []
    for root, _dirs, files in os.walk(mesh_dir):
        for name in sorted(files):
            if name.lower().endswith(MESH_EXTENSIONS):
                paths.append(os.path.join(root, name))
    return paths


# ====================================================================== #
# 新增损坏算子（负样本构造，任务 #32 要求至少两种）
# ====================================================================== #
def corrupt_vertex_noise(
    vertices: Tensor, faces: Tensor, rng: np.random.RandomState, severity: float
) -> Tuple[Tensor, Tensor]:
    """顶点高斯噪声：幅度按包围盒对角线与 severity 缩放。"""
    diag = _bbox_diagonal(vertices)
    sigma = float(severity) * 0.05 * diag
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=tuple(vertices.shape)).astype(np.float32)
    ).to(vertices.device)
    return vertices + noise, faces


def corrupt_face_flip(
    vertices: Tensor, faces: Tensor, rng: np.random.RandomState, severity: float
) -> Tuple[Tensor, Tensor]:
    """随机翻转一部分面片的环绕方向（破坏法线一致性）。"""
    if faces.numel() == 0:
        return vertices, faces
    ratio = min(0.9, 0.1 + 0.8 * float(severity))
    mask = rng.rand(int(faces.shape[0])) < ratio
    flipped = faces.clone()
    flipped[torch.from_numpy(mask)] = faces[torch.from_numpy(mask)][:, [0, 2, 1]]
    return vertices, flipped


def corrupt_decimate(
    vertices: Tensor, faces: Tensor, rng: np.random.RandomState, severity: float
) -> Tuple[Tensor, Tensor]:
    """随机丢弃一部分面片并重映射顶点索引（模拟破洞 / 稀疏化）。"""
    if faces.numel() == 0:
        return vertices, faces
    keep_ratio = max(0.2, 1.0 - 0.7 * float(severity))
    keep = rng.rand(int(faces.shape[0])) < keep_ratio
    if not keep.any():
        keep[0] = True
    kept_faces = faces[torch.from_numpy(keep)]
    # 重映射到被引用的顶点，剔除孤立顶点
    unique_idx, inverse = torch.unique(kept_faces.reshape(-1), return_inverse=True)
    new_faces = inverse.view(kept_faces.shape[0], 3).contiguous()
    return vertices[unique_idx].contiguous(), new_faces


def apply_corruption(
    vertices: Tensor,
    faces: Tensor,
    constructor: AnomalyConstructor,
    rng: np.random.RandomState,
) -> Tuple[Tensor, Tensor, str]:
    """从策略池（旧 4 策略 + 新 3 算子）随机选一种损坏方式施加到 mesh。"""
    strategy = str(rng.choice(np.asarray(constructor.strategies + NEW_CORRUPTION_STRATEGIES)))
    severity = float(rng.uniform(*constructor.severity_range))
    if strategy == "vertex_noise":
        new_v, new_f = corrupt_vertex_noise(vertices, faces, rng, severity)
    elif strategy == "face_flip":
        new_v, new_f = corrupt_face_flip(vertices, faces, rng, severity)
    elif strategy == "decimate":
        new_v, new_f = corrupt_decimate(vertices, faces, rng, severity)
    else:
        new_v, new_f, _info = constructor.construct(vertices, faces, strategy=strategy)
    return new_v, new_f, strategy


# ====================================================================== #
# 特征预计算（CLIP 冻结：渲染 + 编码一次，训练只跑 MLP）
# ====================================================================== #
@torch.no_grad()
def compute_sample_features(
    vertices: Tensor,
    faces: Tensor,
    renderer: MultiViewRenderer,
    clip_encoder: CLIPEncoder,
    geo_dim: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """单个 mesh -> (聚合 CLIP 特征 [clip_dim], 几何描述子 [geo_dim])。

    与 ``train.py:compute_critic_inputs`` / ``inference.py:critic_score_for_mesh``
    完全同一条特征链路：多视图渲染 -> CLIP 逐视图编码（augment=False）->
    视图平均 + re-L2-norm ‖ batch_geometry_features。
    """
    camera_poses, _, _ = renderer.generate_camera_poses(1, device=device)
    rendered = renderer.render(
        vertices.unsqueeze(0).to(device), faces.to(device), camera_poses=camera_poses
    )
    images = rendered["images"]  # [1, N, 3, H, W]
    view_features = clip_encoder.encode_images(
        images.flatten(0, 1), augment=False
    )  # [N, D]
    clip_features = F.normalize(view_features.mean(dim=0, keepdim=True), dim=-1)
    geo_features = batch_geometry_features(
        [vertices.to(device)], [faces.to(device)], geo_dim=geo_dim
    )  # [1, geo_dim]
    return clip_features.cpu()[0], geo_features.cpu()[0]


# ====================================================================== #
# TripoSG 正样本补充（可选；本地跑不了时优雅降级）
# ====================================================================== #
def generate_triposg_positives(
    num: int, config_path: str, device: torch.device
) -> List[Tuple[Tensor, Tensor]]:
    """用 diffusion 后端现采 mesh 作为额外正样本（懒加载，失败即跳过）。"""
    if num <= 0:
        return []
    try:
        from inference import build_diffusion_generator, load_yaml_config

        config = load_yaml_config(config_path)
        generator = build_diffusion_generator(config, device)
    except Exception as exc:
        print(
            f"[警告] TripoSG 后端不可用（{exc}），跳过 --generate_triposg；"
            f"请在服务器上先生成 mesh 再用 --mesh_dir 传入"
        )
        return []

    meshes: List[Tuple[Tensor, Tensor]] = []
    for index in range(int(num)):
        try:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            result = generator.generate(prompt="a 3d object", seed=seed)
            verts = result["vertices"][0].detach().cpu().float()
            faces = result["faces"].detach().cpu().long()
            if faces.dim() == 2 and faces.shape[0] > 0 and verts.shape[0] > 3:
                meshes.append((verts, faces))
                print(f"  TripoSG 正样本 {index + 1}/{num} 生成成功 (seed={seed})")
        except Exception as exc:
            print(f"[警告] TripoSG 样本 {index + 1} 生成失败: {exc}")
    return meshes


# ====================================================================== #
# 训练 / 验证
# ====================================================================== #
def build_feature_pools(
    mesh_paths: List[str],
    extra_meshes: List[Tuple[Tensor, Tensor]],
    val_ratio: float,
    renderer: MultiViewRenderer,
    clip_encoder: CLIPEncoder,
    geo_dim: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Dict[str, Tensor]]:
    """逐 mesh 预计算正/负特征，并按源 mesh 切分 train/val。

    Returns:
        {'train': {'pos_clip','pos_geo','neg_clip','neg_geo'}, 'val': {...}}
        四个张量各自沿样本维堆叠（val 允许为空张量）。
    """
    rng = np.random.RandomState(seed)
    constructor = AnomalyConstructor(seed=seed)

    # 源样本顺序固定后再切分：同一源 mesh 的正负版本只落在同一侧
    num_total = len(mesh_paths) + len(extra_meshes)
    if num_total == 0:
        raise RuntimeError("没有任何正样本来源：--mesh_dir 为空且未启用 --generate_triposg")
    num_val = int(round(num_total * val_ratio))
    num_val = min(num_val, max(num_total - 2, 0))  # 至少保留 2 个训练样本
    order = list(range(num_total))
    rng.shuffle(order)
    val_set = set(order[:num_val])

    pools: Dict[str, Dict[str, List[Tensor]]] = {
        split: {"pos_clip": [], "pos_geo": [], "neg_clip": [], "neg_geo": []}
        for split in ("train", "val")
    }

    def process_one(source_idx: int, vertices: Tensor, faces: Tensor, tag: str) -> None:
        split = "val" if source_idx in val_set else "train"
        clip_pos, geo_pos = compute_sample_features(
            vertices, faces, renderer, clip_encoder, geo_dim, device
        )
        pools[split]["pos_clip"].append(clip_pos)
        pools[split]["pos_geo"].append(geo_pos)

        neg_v, neg_f, strategy = apply_corruption(vertices, faces, constructor, rng)
        clip_neg, geo_neg = compute_sample_features(
            neg_v, neg_f, renderer, clip_encoder, geo_dim, device
        )
        pools[split]["neg_clip"].append(clip_neg)
        pools[split]["neg_geo"].append(geo_neg)
        print(
            f"  [{split}] {tag}: 正例特征 OK；负例策略={strategy}"
        )

    for idx, path in enumerate(mesh_paths):
        loaded = load_mesh_file(path)
        if loaded is None:
            continue
        process_one(idx, loaded[0], loaded[1], os.path.basename(path))
    for j, (verts, faces) in enumerate(extra_meshes):
        process_one(len(mesh_paths) + j, verts, faces, f"triposg_{j}")

    stacked: Dict[str, Dict[str, Tensor]] = {}
    for split in ("train", "val"):
        stacked[split] = {
            key: (torch.stack(tensors, dim=0) if tensors else torch.empty(0))
            for key, tensors in pools[split].items()
        }
    return stacked


def sample_batch(
    pool: Dict[str, Tensor],
    batch_size: int,
    pos_ratio: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """从特征池有放回采样一个正负混合 batch。

    Returns:
        (clip [B, D], geo [B, G], labels [B])
    """
    num_pos = int(round(batch_size * pos_ratio))
    num_pos = min(max(num_pos, 0), batch_size)
    num_neg = batch_size - num_pos

    n_pos = int(pool["pos_clip"].shape[0])
    n_neg = int(pool["neg_clip"].shape[0])
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("特征池中正例或负例为空，无法构造训练 batch")

    pos_idx = torch.randint(n_pos, (num_pos,))
    neg_idx = torch.randint(n_neg, (num_neg,))

    clip = torch.cat([pool["pos_clip"][pos_idx], pool["neg_clip"][neg_idx]], dim=0)
    geo = torch.cat([pool["pos_geo"][pos_idx], pool["neg_geo"][neg_idx]], dim=0)
    labels = torch.cat(
        [torch.ones(num_pos), torch.zeros(num_neg)], dim=0
    )
    # 打乱正负顺序，避免模型学到「前半全是正例」的平凡规律
    perm = torch.randperm(batch_size)
    return clip[perm], geo[perm], labels[perm]


@torch.no_grad()
def evaluate_pool(critic: SemanticCritic, pool: Dict[str, Tensor]) -> Optional[float]:
    """在特征池上计算 accuracy；池为空返回 None。"""
    if pool["pos_clip"].numel() == 0 or pool["neg_clip"].numel() == 0:
        return None
    device = next(critic.parameters()).device  # 特征池存 CPU，前向前搬到 critic 设备
    clip = torch.cat([pool["pos_clip"], pool["neg_clip"]], dim=0).to(device)
    geo = torch.cat([pool["pos_geo"], pool["neg_geo"]], dim=0).to(device)
    labels = torch.cat(
        [
            torch.ones(pool["pos_clip"].shape[0]),
            torch.zeros(pool["neg_clip"].shape[0]),
        ],
        dim=0,
    )
    probs = critic(clip, geo).squeeze(-1)  # [N]
    labels = labels.to(probs.device)
    preds = (probs > 0.5).float()
    return float((preds == labels).float().mean().item())


# ====================================================================== #
# 主流程
# ====================================================================== #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SemanticCritic 继续教育（Learned Semantic Reward 阶段一）"
    )
    parser.add_argument(
        "--mesh_dir", type=str, default=None, help="正例 mesh 目录（obj/glb 混合，递归扫描）"
    )
    parser.add_argument(
        "--generate_triposg",
        type=int,
        default=0,
        help="用本地 TripoSG 现采的正例数量（可选；本地不可用时自动跳过）",
    )
    parser.add_argument(
        "--triposg_config",
        type=str,
        default="configs/diffusion_inference.yaml",
        help="--generate_triposg > 0 时使用的 diffusion 推理配置",
    )
    parser.add_argument(
        "--old_critic",
        type=str,
        required=True,
        help="旧 Critic checkpoint（train.py 保存格式，含 'critic' / 'config' 键）",
    )
    parser.add_argument("--out", type=str, default="outputs/critic_v2.pt")
    parser.add_argument("--steps", type=int, default=3000, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--pos_ratio", type=float, default=0.5, help="每 batch 正例占比（0.5 = 1:1）"
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="按源 mesh 留出的验证比例")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动")
    parser.add_argument("--log_interval", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"设备: {device}")

    # ---- 旧 Critic：权重 + 训练渲染配置一并继承 ---- #
    critic, geo_dim, render_cfg = build_critic(args.old_critic, device)
    # build_critic 返回的是推理态（eval + 全参数冻结）；继续教育需要解冻 MLP
    critic.train()
    for param in critic.parameters():
        param.requires_grad_(True)
    old_state = torch.load(args.old_critic, map_location="cpu", weights_only=False)
    old_config: Dict[str, Any] = dict(old_state.get("config", {}) or {})
    critic_cfg = dict(old_config.get("semantic_critic", {}) or {})
    vlm_cfg = dict(old_config.get("vlm", {}) or {})
    clip_dim = int(critic_cfg.get("clip_dim", vlm_cfg.get("clip_dim", 512)))
    hidden_dim = int(critic_cfg.get("hidden_dim", 512))
    dropout = float(critic_cfg.get("dropout", 0.1))
    print(
        f"旧 Critic 已加载: clip_dim={clip_dim}, geo_dim={geo_dim}, "
        f"hidden={hidden_dim}, dropout={dropout}"
    )

    # 渲染配置严格沿用旧 Critic 训练时的设置（默认 256 / 4 / 2.5 / ±30）
    elevation = tuple(render_cfg.get("elevation_range", (-30.0, 30.0)))
    renderer = MultiViewRenderer(
        image_size=int(render_cfg.get("image_size", 256)),
        num_views=int(render_cfg.get("num_views", 4)),
        camera_distance=float(render_cfg.get("camera_distance", 2.5)),
        elevation_range=(float(elevation[0]), float(elevation[1])),
        azimuth_strategy=str(render_cfg.get("azimuth_strategy", "stratified")),
        device=str(device),
    )
    print(
        f"Critic 训练渲染配置: {renderer.image_size}px × {renderer.num_views} 视图, "
        f"cam={renderer.camera_distance}, el={renderer.elevation_range}, "
        f"azimuth={renderer.azimuth_strategy}"
    )

    # ---- 冻结 CLIP（特征离线预计算，训练期完全不碰 CLIP）---- #
    clip_encoder = CLIPEncoder(
        model_name=str(vlm_cfg.get("model_name", "ViT-B/32")),
        device=str(device),
        input_range="zero_one",
        pretrained=str(vlm_cfg.get("pretrained", "openai")),
    )
    clip_encoder.set_training_mode(False)

    # ---- 正样本来源 ---- #
    mesh_paths: List[str] = []
    if args.mesh_dir:
        if not os.path.isdir(args.mesh_dir):
            print(f"[错误] --mesh_dir 不存在: {args.mesh_dir}", file=sys.stderr)
            return 1
        mesh_paths = scan_mesh_dir(args.mesh_dir)
        print(f"mesh 目录扫描到 {len(mesh_paths)} 个文件: {args.mesh_dir}")
    extra_meshes = generate_triposg_positives(
        args.generate_triposg, args.triposg_config, device
    )

    # ---- 特征预计算（正/负一次性产出，10% 源 mesh 留作验证）---- #
    print("开始特征预计算（渲染 + CLIP + 几何描述子）...")
    pools = build_feature_pools(
        mesh_paths,
        extra_meshes,
        args.val_ratio,
        renderer,
        clip_encoder,
        geo_dim,
        device,
        args.seed,
    )
    num_train_pos = int(pools["train"]["pos_clip"].shape[0])
    num_val_pos = int(pools["val"]["pos_clip"].shape[0])
    if num_train_pos == 0:
        print("[错误] 训练集为空，请检查 --mesh_dir 内容", file=sys.stderr)
        return 1
    print(f"特征池: train 正例 {num_train_pos} / 负例 {num_train_pos}，"
          f"val 正例 {num_val_pos} / 负例 {num_val_pos}")

    # ---- 只训 Critic MLP（spectral-norm 参数），BCE loss ---- #
    critic.train()
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    pos_ratio = min(max(float(args.pos_ratio), 0.0), 1.0)
    for step in range(1, int(args.steps) + 1):
        clip_batch, geo_batch, labels = sample_batch(
            pools["train"], args.batch_size, pos_ratio
        )
        clip_batch = clip_batch.to(device)
        geo_batch = geo_batch.to(device)
        labels = labels.to(device)

        logits = critic(clip_batch, geo_batch, return_logits=True).squeeze(-1)
        loss = bce(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % max(args.log_interval, 1) == 0 or step == 1:
            acc = float(((torch.sigmoid(logits) > 0.5).float() == labels).float().mean())
            print(f"step {step}/{args.steps} | loss {loss.item():.4f} | batch_acc {acc:.3f}")

    # ---- 结束时打印 train / val accuracy ---- #
    critic.eval()
    train_acc = evaluate_pool(critic, pools["train"])
    val_acc = evaluate_pool(critic, pools["val"])
    print(
        f"训练结束 | train_accuracy={train_acc:.4f}"
        + (f" | val_accuracy={val_acc:.4f}" if val_acc is not None else " | val 为空")
    )

    # ---- 保存：与旧 checkpoint 完全一致的格式（build_critic 原样加载）---- #
    new_config: Dict[str, Any] = {
        "semantic_critic": {
            "enabled": True,
            "clip_dim": clip_dim,
            "geo_dim": geo_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        },
        "rendering": {
            "image_size": int(render_cfg.get("image_size", 256)),
            "num_views": int(render_cfg.get("num_views", 4)),
            "camera_distance": float(render_cfg.get("camera_distance", 2.5)),
            "elevation_range": [float(elevation[0]), float(elevation[1])],
            "azimuth_strategy": str(render_cfg.get("azimuth_strategy", "stratified")),
        },
        "vlm": dict(vlm_cfg),
        "finetune": {
            "source": "finetune_critic.py (Learned Semantic Reward 阶段一)",
            "old_critic": os.path.abspath(args.old_critic),
            "steps": int(args.steps),
            "pos_ratio": pos_ratio,
            "num_train_samples": num_train_pos,
            "num_val_samples": num_val_pos,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save({"critic": critic.state_dict(), "config": new_config}, args.out)
    print(f"critic_v2 已保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

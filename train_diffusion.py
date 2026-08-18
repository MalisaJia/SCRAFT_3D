"""TripoSG LoRA fine-tuning on precomputed Objaverse latents.

Usage:
    python train_diffusion.py --config configs/train_diffusion.yaml [--resume checkpoint.pt]

单网络扩散训练管线（与 GAN 的 ``train.py`` 完全独立，不共享任何训练状态）：
1. 加载冻结的 TripoSG DiT（``TripoSGDiTModel``，diffusers 目录布局权重），
   对注意力 ``to_q`` / ``to_v`` 投影注入 LoRA 适配器（仅训练 LoRA 参数）
2. Rectified flow 目标：``x_t = (1-t)*x_0 + t*noise``，预测速度 ``v = noise - x_0``，
   MSE 损失；时间步按真实调度器约定缩放至 [0, 1000]（``training.timestep_scale``）
3. 数据来自 ``scripts/precompute_latents.py`` 预计算的 VAE latent shard
   （latent 形状 [2048, 64]），训练循环内不做任何渲染 / VAE 编解码
   （纯张量运算，速度最大化）；shard 中若含 DINOv2 ``image_embeds`` 则作为
   cross-attention 条件，缺失时按无条件（零嵌入，与官方 CFG 空分支一致）训练
4. 训练特性：bf16 autocast、梯度检查点、LoRA 权重 EMA、cosine LR + warmup
5. 定期评估：用 EMA 权重 Euler 采样 latent -> VAE 解码 SDF ->
   ``triposg.inference_utils.hierarchical_extract_geometry`` 八叉树提取
   -> ``MultiViewRenderer`` 渲染预览图（渲染参数取自配置 ``rendering`` 段；
   评估失败不影响训练），并在依赖可用时附加 CLIP Score / Critic Score
6. 可选 held-out 验证：从 latent 缓存尾部切出 ``evaluation.num_val_samples``
   个样本（永不参与训练），每 ``evaluation.interval`` 步计算一次无梯度 MSE
7. 日志：TensorBoard（loss / LR / grad_norm / 评估渲染图 / ``eval/*`` 指标）
   + 文本日志，风格与 ``train.py`` 保持一致
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# TensorBoard 为可选依赖：缺失时训练照常进行，只是不写标量日志
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover - 取决于运行环境是否安装 tensorboard
    HAS_TENSORBOARD = False

# 允许从任意工作目录运行本脚本（把项目根目录加入 import 路径）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LOGGER = logging.getLogger("diffusion_train")

# 梯度裁剪阈值（LoRA 微调通常较稳定，1.0 足以防御偶发尖峰）
GRAD_CLIP = 1.0

# 评估时 Euler 采样的步数（预览用途，不必追求高质量）
EVAL_NUM_STEPS = 25

# 评估时 VAE 解码 SDF 的八叉树深度（res = 2^depth；8 -> 256，与正式推理
# configs/diffusion_inference.yaml 的 octree_resolution=256 对齐）
EVAL_OCTREE_DEPTH = 8

# 评估采样的 classifier-free guidance 强度（与 TripoSG 官方推理默认值一致；
# 基座在推理期依赖 CFG 放大条件方向，不加引导会偏离数据流形出碎片几何）
EVAL_GUIDANCE_SCALE = 7.0

# 真实 DiT 的 cross-attention 条件维度（DINOv2 large = 1024）
DIT_COND_DIM = 1024

# DINOv2 图像条件的 token 数（仅作 shard 缺 image_embeds 时的兜底默认值；正常
# 运行由数据集探测首个 shard 的实际形状）。官方 feature_extractor_dinov2 为
# 224/center-crop，patch_size=14 -> (224/14)^2 = 256 patches + 1 CLS = 257
# （官方 CFG 空分支为 torch.zeros_like(image_embeds)，形状同 [B, 257, 1024]）
DIT_COND_NUM_TOKENS = 257

# 每次评估渲染的样本数
EVAL_NUM_SAMPLES = 4

# 验证损失的固定噪声种子：每次验证复用同一批噪声 / 时间步，使不同步数之间的
# 验证 MSE 可直接比较（否则 rectified flow 的随机 t 会带来很大方差）
VAL_NOISE_SEED = 1234

# 评估用 CLIP / Critic 的进程内缓存（键 -> 模型或 None）。这两个模型只在评估时
# 用到，但重复加载很慢，因此加载一次后常驻；加载失败也记入缓存，避免每轮重试
_EVAL_MODEL_CACHE: Dict[str, Any] = {}


# ====================================================================== #
# 配置
# ====================================================================== #
def load_config(path: str) -> Dict[str, Any]:
    """加载 YAML 训练配置（缺失的顶层 block 补成空 dict）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    # rendering / evaluation 为后加入的可选段，旧配置缺失时补空 dict，
    # 读取方一律带默认值回落，训练行为与原来完全一致
    for block in ("model", "training", "data", "logging", "rendering", "evaluation"):
        config.setdefault(block, {})
    return config


def setup_logging(output_dir: str) -> None:
    """同时输出到控制台和 ``<output_dir>/train.log``（与 train.py 一致）。"""
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)

    file_handler = logging.FileHandler(
        os.path.join(output_dir, "train.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def set_seed(seed: int) -> None:
    """固定随机种子（不启用 deterministic，以免明显变慢）。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ====================================================================== #
# LoRA 适配器
# ====================================================================== #
class LoRALinear(nn.Module):
    """低秩适配的 Linear：``y = W x + scale * B A x``。

    原始权重全程冻结（requires_grad=False），只训练低秩分支 A / B；
    B 初始化为零，保证注入瞬间模型行为与预训练完全一致。
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / float(rank)

        in_features = base.in_features
        out_features = base.out_features
        # 与主干层同设备 / 同 dtype 分配，避免 DiT 在 CUDA 上时设备不匹配崩溃
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # 冻结主干权重，优化器只收集 LoRA 参数
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


# 目标模块名的常见别名：真实 TripoSG DiT 用 diffusers ``Attention``，
# 投影层命名为 ``to_q`` / ``to_k`` / ``to_v`` / ``to_out``；保留旧别名兼容
_TARGET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "q_proj": ("to_q", "q_proj", "query"),
    "v_proj": ("to_v", "v_proj", "value"),
    "k_proj": ("to_k", "k_proj", "key"),
    "o_proj": ("to_out", "o_proj", "proj"),
    "to_q": ("to_q",),
    "to_v": ("to_v",),
    "to_k": ("to_k",),
}


def _match_target(module_name: str, targets: List[str]) -> bool:
    """判断模块名是否命中任一目标（含别名展开）。

    两种匹配模式：

    - **带点目标**（如 ``"attn2.to_q"``）：对完整模块名做**后缀匹配**
      （``module_name == target`` 或以 ``"." + target`` 结尾），用于限定
      注意力类型，例如只给 cross-attention（attn2）挂适配器；
    - **裸名目标**（如 ``"to_q"``）：保持历史行为不变 —— 别名展开后
      对叶子名做相等或子串匹配。
    """
    leaf = module_name.rsplit(".", 1)[-1] if "." in module_name else module_name
    for target in targets:
        if "." in target:
            if module_name == target or module_name.endswith("." + target):
                return True
            continue
        for alias in _TARGET_ALIASES.get(target, (target,)):
            if alias == leaf or alias in leaf:
                return True
    return False


def inject_lora(
    dit: nn.Module, rank: int, target_modules: List[str], alpha: Optional[float] = None
) -> Dict[str, LoRALinear]:
    """给 DiT 的注意力投影注入 LoRA 适配器（原地替换 Linear 模块）。

    真实 TripoSG DiT（``TripoSGDiTModel``）的自注意力 / 交叉注意力投影位于
    ``blocks.{i}.attn1.{to_q,to_v}`` 与 ``blocks.{i}.attn2.{to_q,to_v}``
    （diffusers ``Attention`` 模块）。目标名支持两种写法：

    - 裸名 ``["to_q", "to_v"]``：子串匹配叶子名，自注意力 + 交叉注意力全挂；
    - 带点模式 ``["attn2.to_q", "attn2.to_v"]``：完整模块名后缀匹配，
      仅挂 cross-attention（挂点数减半）。

    Args:
        dit: 冻结的 TripoSG DiT。
        rank: LoRA 秩。
        target_modules: 目标模块名列表（裸名 ``["to_q", "to_v"]`` 或带点
            后缀模式 ``["attn2.to_q", "attn2.to_v"]``）。
        alpha: LoRA 缩放系数，默认取 rank（scaling=1）。

    Returns:
        注入后的 ``{完整模块名: LoRALinear}`` 字典；为空说明未命中任何层。
    """
    alpha = float(alpha if alpha is not None else rank)
    injected: Dict[str, LoRALinear] = {}

    for name, module in list(dit.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if not _match_target(name, target_modules):
            continue
        # 找到父模块并原地替换属性
        parent = dit
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            parent = dit.get_submodule(parent_name)
        adapter = LoRALinear(module, rank, alpha)
        setattr(parent, leaf, adapter)
        injected[name] = adapter

    if not injected:
        raise RuntimeError(
            f"未在 DiT 中找到任何匹配 {target_modules} 的 Linear 层，"
            "请检查 model.diffusion.lora_target_modules 配置"
        )
    return injected


def lora_state_dict(adapters: Dict[str, LoRALinear]) -> Dict[str, Tensor]:
    """只提取 LoRA 分支的权重（A / B），checkpoint 体积与秩成正比。"""
    state: Dict[str, Tensor] = {}
    for name, adapter in adapters.items():
        state[f"{name}.lora_A"] = adapter.lora_A.detach().clone()
        state[f"{name}.lora_B"] = adapter.lora_B.detach().clone()
    return state


def load_lora_state_dict(adapters: Dict[str, LoRALinear], state: Dict[str, Tensor]) -> None:
    """把 checkpoint 中的 LoRA 权重写回适配器。

    双向校验：适配器缺键会 KeyError（原有行为）；权重里存在未被任何已挂载
    适配器消费的键同样 KeyError（防止训练 84 挂点权重被推理 42 挂点加载时
    attn1 增量被静默丢弃）。
    """
    for name, adapter in adapters.items():
        key_a, key_b = f"{name}.lora_A", f"{name}.lora_B"
        if key_a not in state or key_b not in state:
            raise KeyError(f"checkpoint 缺少 LoRA 权重: {key_a} / {key_b}")
        adapter.lora_A.data.copy_(state[key_a].to(adapter.lora_A.device))
        adapter.lora_B.data.copy_(state[key_b].to(adapter.lora_B.device))

    consumed = {
        f"{name}.{suffix}" for name in adapters for suffix in ("lora_A", "lora_B")
    }
    unconsumed = [key for key in state if key not in consumed]
    if unconsumed:
        preview = ", ".join(unconsumed[:4])
        raise KeyError(
            f"LoRA 权重含 {len(unconsumed)} 个未被任何已挂载适配器消费的键"
            f"（前几个: {preview} ...）。通常是挂点数少于训练侧（如 v4 的 42 挂点"
            "加载旧 84 挂点权重），请核对 lora_target_modules 与训练配置一致"
        )


# 优化器默认 weight decay（历史硬编码值；lora_B 单独分组时 lora_A 沿用该值）
DEFAULT_WEIGHT_DECAY = 0.01


def build_lora_param_groups(
    adapters: Dict[str, LoRALinear], lora_b_weight_decay: float = 0.0
) -> Any:
    """构造优化器参数（B 矩阵锚：lora_B 可单独施加 weight_decay）。

    - ``lora_b_weight_decay <= 0.0``（配置缺省）：返回与历史行为一致的扁平
      参数列表（逐适配器 A / B 交错排列，单一默认参数组），训练逐位不变；
    - 否则返回两个参数组：lora_B 组施加 ``lora_b_weight_decay``（把 B 矩阵
      锚向零，抑制 LoRA 增量幅度失控），lora_A 组保持 ``DEFAULT_WEIGHT_DECAY``。
    """
    if float(lora_b_weight_decay) <= 0.0:
        return [
            param
            for adapter in adapters.values()
            for param in (adapter.lora_A, adapter.lora_B)
        ]
    group_b = {
        "params": [adapter.lora_B for adapter in adapters.values()],
        "weight_decay": float(lora_b_weight_decay),
    }
    group_a = {
        "params": [adapter.lora_A for adapter in adapters.values()],
        "weight_decay": DEFAULT_WEIGHT_DECAY,
    }
    return [group_b, group_a]


# ====================================================================== #
# TripoSG 加载（懒加载，与 diffusion_adapter.py 同一模式）
# ====================================================================== #
def _resolve_weights_dir(weights_path: str) -> Optional[Path]:
    """解析 weights_path：文件 -> 其所在目录；目录 -> 本身；不存在 -> None。"""
    if not weights_path:
        return None
    path = Path(weights_path)
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    return None


def build_dit_with_lora(
    config: Dict[str, Any], device: torch.device
) -> Tuple[nn.Module, Dict[str, LoRALinear], Optional[Any]]:
    """加载冻结的 TripoSG DiT（``TripoSGDiTModel``），注入 LoRA 适配器。

    权重为 diffusers 目录布局（``model_index.json`` + ``transformer/`` +
    ``vae/`` 等子目录），通过 ``from_pretrained(..., subfolder=...)`` 装配；
    本地目录缺少 ``model_index.json`` 时回退 HuggingFace ``VAST-AI/TripoSG``。

    Args:
        config: 完整训练配置（读取 ``model.diffusion`` 段）。
        device: 模型设备。

    Returns:
        (dit, lora_adapters, vae_or_none)：DiT（主干冻结）、LoRA 适配器字典、
        VAE 实例（加载成功时返回，供评估解码使用；否则为 None）。

    Raises:
        RuntimeError: TripoSG 未安装或权重加载失败。
    """
    diffusion_cfg = (config.get("model", {}) or {}).get("diffusion", {})
    weights_path = str(diffusion_cfg.get("weights_path", "") or "")
    rank = int(diffusion_cfg.get("lora_rank", 16))
    targets = list(diffusion_cfg.get("lora_target_modules", ["to_q", "to_v"]))

    try:
        from triposg.models.autoencoders import TripoSGVAEModel
        from triposg.models.transformers import TripoSGDiTModel
    except ImportError as exc:
        raise RuntimeError(
            "未能导入 TripoSG（triposg.models）。"
            "请先克隆 TripoSG 仓库并安装其依赖，或检查 PYTHONPATH。"
        ) from exc

    dit = None
    vae = None
    weights_dir = _resolve_weights_dir(weights_path)
    if weights_dir is not None and (weights_dir / "model_index.json").is_file():
        try:
            dit = TripoSGDiTModel.from_pretrained(str(weights_dir), subfolder="transformer")
        except Exception as exc:
            LOGGER.warning("本地 DiT 加载失败（%s），回退 HuggingFace", exc)
        vae_dir = weights_dir / "vae"
        if (vae_dir / "config.json").is_file():
            try:
                vae = TripoSGVAEModel.from_pretrained(str(weights_dir), subfolder="vae")
            except Exception as exc:  # 评估解码非必需，失败只警告不阻断
                LOGGER.warning("本地 VAE 加载失败（%s），评估阶段将跳过 mesh 解码", exc)
                vae = None
        else:
            LOGGER.warning("未找到 VAE 权重目录 %s，评估阶段将跳过 mesh 解码", vae_dir)

    if dit is None:
        # 本地权重不可用时回退 HuggingFace 官方仓库（VAST-AI/TripoSG）
        from triposg.pipelines import TripoSGPipeline

        LOGGER.info("从 HuggingFace VAST-AI/TripoSG 加载权重...")
        pipeline = TripoSGPipeline.from_pretrained("VAST-AI/TripoSG")
        dit = pipeline.transformer
        if vae is None:
            vae = pipeline.vae

    dit = dit.to(device)
    if vae is not None:
        vae = vae.to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad_(False)

    # ---- 冻结主干 ----
    dit.eval()
    for param in dit.parameters():
        param.requires_grad_(False)

    # ---- 注入 LoRA（注入后切回 train 模式，仅 LoRA 分支参与训练）----
    adapters = inject_lora(dit, rank, targets)
    dit.train()
    num_lora_params = sum(
        adapter.lora_A.numel() + adapter.lora_B.numel()
        for adapter in adapters.values()
    )
    LOGGER.info(
        "LoRA 注入完成：%d 个注意力投影层，rank=%d，可训练参数 %.2fM",
        len(adapters),
        rank,
        num_lora_params / 1e6,
    )
    return dit, adapters, vae


# ====================================================================== #
# DiT 前向（真实签名：hidden_states / timestep / encoder_hidden_states）
# ====================================================================== #
def forward_dit(dit: nn.Module, x_t: Tensor, t: Tensor, cond: Optional[Tensor] = None) -> Tensor:
    """调用 TripoSG DiT 预测速度场。

    真实 ``TripoSGDiTModel.forward`` 签名::

        forward(hidden_states, timestep, encoder_hidden_states=None, ...)
        -> Transformer1DModelOutput(sample=[B, N, 64])

    Args:
        dit: TripoSG DiT（含 LoRA 适配器）。
        x_t: 加噪 latent，[B, N, 64]。
        t: 时间步，[B]（按调度器约定缩放到 [0, 1000]）。
        cond: DINOv2 图像条件嵌入 [B, S, 1024]；None 表示无条件（与官方
            CFG 的零嵌入空分支等价，DiT 内部对 None 同样处理）。

    Returns:
        速度预测 [B, N, 64]（与 x_t 同形状）。
    """
    output = dit(
        hidden_states=x_t,
        timestep=t,
        encoder_hidden_states=cond,
        return_dict=False,
    )
    return output[0]


# ====================================================================== #
# 梯度检查点
# ====================================================================== #
def enable_gradient_checkpointing(dit: nn.Module) -> int:
    """对 DiT 中的 transformer block 序列启用梯度检查点。

    扫描所有长度 >= 2 的 ``nn.ModuleList``（典型的 blocks / layers 容器），
    把每个 block 的 forward 包装为 ``torch.utils.checkpoint.checkpoint``。

    Returns:
        被包装的 block 数量（0 表示未找到可包装的序列）。
    """
    from torch.utils.checkpoint import checkpoint

    wrapped = 0
    for module in dit.modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        for i, block in enumerate(module):
            if not isinstance(block, nn.Module):
                continue
            original_forward = block.forward

            def checkpointed(*args: Any, _fwd: Any = original_forward, **kwargs: Any) -> Tensor:
                return checkpoint(_fwd, *args, use_reentrant=False, **kwargs)

            block.forward = checkpointed  # type: ignore[method-assign]
            wrapped += 1
    return wrapped


# ====================================================================== #
# Latent 缓存数据集
# ====================================================================== #
class LatentCacheDataset(Dataset):
    """读取 ``scripts/precompute_latents.py`` 输出的 latent shard 数据集。

    shard 文件为 ``latent_shard_{i}.pt``，内含 ``{'latents', 'captions', 'uids'}``；
    ``manifest.json`` 记录各 shard 的样本数。shard 张量按需加载并缓存在进程内
    （多 worker 下每个 worker 独立缓存各自访问过的 shard）。

    可选传入 ``critic_weights_path``（Learned Semantic Reward 阶段二的
    ``critic_weights.json``，``{uid: weight}``）：每个样本附带其训练权重，
    缺失 uid 一律按 1.0 处理，不改变其他训练逻辑。
    """

    def __init__(self, cache_dir: str, critic_weights_path: Optional[str] = None) -> None:
        self.cache_dir = cache_dir

        # Learned Semantic Reward 训练加权：uid -> 权重（缺失样本回落 1.0）
        self.critic_weights: Dict[str, float] = {}
        if critic_weights_path:
            if not os.path.isfile(critic_weights_path):
                raise FileNotFoundError(
                    f"critic 权重文件不存在: {critic_weights_path}，"
                    f"请先运行 scripts/score_latents_critic.py"
                )
            with open(critic_weights_path, "r", encoding="utf-8") as handle:
                self.critic_weights = {
                    str(uid): float(weight)
                    for uid, weight in json.load(handle).items()
                }
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"找不到 {manifest_path}，请先运行 scripts/precompute_latents.py"
            )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        # 展平索引：sample_index -> (shard 文件名, shard 内下标)
        self.index: List[Tuple[str, int]] = []
        for shard_info in self.manifest.get("shards", []):
            shard_path = os.path.join(cache_dir, shard_info["path"])
            if not os.path.isfile(shard_path):
                LOGGER.warning("manifest 中的 shard 不存在，跳过: %s", shard_path)
                continue
            n = int(shard_info.get("num_samples", 0))
            self.index.extend((shard_info["path"], i) for i in range(n))

        if not self.index:
            raise RuntimeError(f"latent 缓存为空: {cache_dir}")
        self._shard_cache: Dict[str, Dict[str, Any]] = {}

        # 扫描全部 shard 的 image_embeds token 维（混装护栏）：不同口径
        # （如 1370 与 257）混在同一缓存里会在 DataLoader collate 时
        # 训练中途崩溃，这里初始化时全量扫描、发现不一致立即报错。
        # shard 不大，逐个加载可接受；扫完释放进程缓存避免常驻内存
        shard_names: List[str] = []
        for shard_name, _ in self.index:
            if shard_name not in shard_names:
                shard_names.append(shard_name)
        embed_shapes: Dict[Tuple[int, int], List[str]] = {}
        for shard_name in shard_names:
            embeds = self._get_shard(shard_name).get("image_embeds", None)
            if embeds is None:
                continue  # 无 image_embeds 的 shard 按零嵌入处理，不参与形状校验
            embed_shapes.setdefault(
                (int(embeds.shape[1]), int(embeds.shape[2])), []
            ).append(shard_name)
        self._shard_cache.clear()
        if len(embed_shapes) > 1:
            detail = "; ".join(
                f"{tokens}x{dim} token: {names}"
                for (tokens, dim), names in embed_shapes.items()
            )
            raise RuntimeError(
                f"latent 缓存 {cache_dir} 的 image_embeds 形状混装：{detail}。"
                "不同 token 口径（如 1370 / 257）禁止混入同一训练缓存，"
                "请重建缓存（scripts/precompute_latents.py / precompute_objaverse_stream.py）"
            )
        if embed_shapes:
            # 形状已验证全局一致，任取一组作为条件嵌入形状
            (self.cond_num_tokens, self.cond_dim) = next(iter(embed_shapes))
        else:
            # 整个缓存都没有 image_embeds：用默认值造零嵌入
            self.cond_num_tokens = DIT_COND_NUM_TOKENS
            self.cond_dim = DIT_COND_DIM

    def __len__(self) -> int:
        return len(self.index)

    def _get_shard(self, shard_name: str) -> Dict[str, Any]:
        """加载（或从进程内缓存取）shard 张量。"""
        if shard_name not in self._shard_cache:
            self._shard_cache[shard_name] = torch.load(
                os.path.join(self.cache_dir, shard_name), map_location="cpu"
            )
        return self._shard_cache[shard_name]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_name, inner_idx = self.index[idx]
        shard = self._get_shard(shard_name)
        latent = shard["latents"][inner_idx].float()  # fp16 -> fp32
        captions = shard.get("captions", [])
        caption = captions[inner_idx] if inner_idx < len(captions) else ""
        uids = shard.get("uids", [])
        uid = str(uids[inner_idx]) if inner_idx < len(uids) else f"{shard_name}:{inner_idx}"
        # Learned Semantic Reward 训练加权：缺失 uid 按 1.0（不改变原始损失）
        weight = float(self.critic_weights.get(uid, 1.0)) if self.critic_weights else 1.0
        # DINOv2 图像条件嵌入（可选）：缺失时填零（与官方 CFG 空分支一致）
        image_embeds = shard.get("image_embeds", None)
        if image_embeds is not None:
            image_embeds = image_embeds[inner_idx].float()
        else:
            image_embeds = torch.zeros(
                (self.cond_num_tokens, self.cond_dim), dtype=torch.float32
            )
        return {
            "latent": latent,
            "caption": caption,
            "image_embeds": image_embeds,
            "uid": uid,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


def build_dataloader(config: Dict[str, Any], dataset: LatentCacheDataset) -> DataLoader:
    """构建 latent 训练 dataloader（固定形状张量，直接默认 collate）。"""
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def infinite_batches(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    """无限循环迭代 dataloader（每个 epoch 自动重新洗牌）。"""
    while True:
        for batch in loader:
            yield batch


def _dataset_uid_weighted(
    dataset: LatentCacheDataset, shard_name: str, inner_idx: int
) -> bool:
    """判断某条样本的 uid 是否在 Critic 权重表内（启动期覆盖率统计用）。"""
    shard = dataset._get_shard(shard_name)
    uids = shard.get("uids", [])
    uid = str(uids[inner_idx]) if inner_idx < len(uids) else f"{shard_name}:{inner_idx}"
    return uid in dataset.critic_weights


def split_validation_batches(
    dataset: LatentCacheDataset,
    num_samples: int,
    batch_size: int,
    min_train_samples: int = 1,
) -> List[Dict[str, Tensor]]:
    """从 latent 缓存尾部切出 held-out 验证集（就地从训练索引中移除）。

    ``dataset.index`` 按 shard 顺序展平，取末尾 ``num_samples`` 条等于取最后一个
    shard 的尾部样本；预取成 CPU 张量 batch 列表后从训练索引删除，保证验证
    样本永不参与训练。必须在构建 DataLoader **之前**调用，否则 worker 进程拿到的
    是切分前的索引副本。

    Args:
        dataset: latent 缓存数据集（就地修改 ``index``）。
        num_samples: 验证样本数；<= 0 时不切分。
        batch_size: 验证 batch 大小。
        min_train_samples: 切分后至少需留给训练的样本数（传训练 batch_size），
            不够时自动缩小甚至放弃切分。

    Returns:
        验证 batch 列表，每项为 ``{'latent': [B, N, 64], 'image_embeds': [B, S, D]}``
        （fp16 常驻内存，与 latent 缓存本身的存储精度一致，前向时再转 fp32）；
        未切分时返回空列表。
    """
    total = len(dataset.index)
    num_samples = min(int(num_samples), max(total - max(1, int(min_train_samples)), 0))
    if num_samples <= 0:
        return []

    start_index = total - num_samples
    items = [dataset[start_index + i] for i in range(num_samples)]
    del dataset.index[start_index:]
    # 预取会把尾部 shard 读进进程缓存，取完立即释放内存
    dataset._shard_cache.clear()

    batch_size = max(1, int(batch_size))
    batches: List[Dict[str, Tensor]] = []
    for start in range(0, num_samples, batch_size):
        chunk = items[start : start + batch_size]
        batches.append(
            {
                "latent": torch.stack([item["latent"] for item in chunk]).half(),
                "image_embeds": torch.stack(
                    [item["image_embeds"] for item in chunk]
                ).half(),
            }
        )
    return batches


# ====================================================================== #
# LR 调度器（cosine + warmup）
# ====================================================================== #
def build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, max_iterations: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """线性 warmup 后 cosine 衰减到 0。"""
    warmup_steps = max(1, int(warmup_steps))
    max_iterations = max(warmup_steps + 1, int(max_iterations))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max_iterations - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ====================================================================== #
# EMA（仅针对 LoRA 权重）
# ====================================================================== #
class LoRAEMA:
    """LoRA 权重的指数滑动平均（评估 / 导出用）。"""

    def __init__(self, adapters: Dict[str, LoRALinear], decay: float) -> None:
        self.decay = float(decay)
        self.shadow = lora_state_dict(adapters)

    @torch.no_grad()
    def update(self, adapters: Dict[str, LoRALinear]) -> None:
        state = lora_state_dict(adapters)
        for key, value in self.shadow.items():
            value.mul_(self.decay).add_(state[key].to(value.device), alpha=1.0 - self.decay)

    def apply(self, adapters: Dict[str, LoRALinear]) -> Dict[str, Tensor]:
        """把 EMA 权重临时写入适配器，返回原权重用于之后恢复。"""
        backup = lora_state_dict(adapters)
        load_lora_state_dict(adapters, self.shadow)
        return backup

    @staticmethod
    def restore(adapters: Dict[str, LoRALinear], backup: Dict[str, Tensor]) -> None:
        """恢复 ``apply`` 前的原始权重。"""
        load_lora_state_dict(adapters, backup)


# ====================================================================== #
# 训练步：rectified flow
# ====================================================================== #
def parse_condition_dropout(train_cfg: Dict[str, Any]) -> float:
    """读取 ``training.condition_dropout``（默认 0.0 = 不启用）并校验区间。

    合法值必须在 [0, 1]；越界或 NaN 一律 ValueError（NaN 的比较恒为 False，
    会被同一条件拦下），防止坏配置静默进入训练循环。
    """
    value = float(train_cfg.get("condition_dropout", 0.0))
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"training.condition_dropout 必须在 [0, 1]，实际 {value}"
        )
    return value


def parse_lora_b_weight_decay(train_cfg: Dict[str, Any]) -> float:
    """读取 ``training.lora_b_weight_decay``（默认 0.0 = 关闭）并校验。

    必须为非负有限值；负值 / NaN / inf 一律 ValueError，防止坏配置静默进入
    优化器。0.0 = 与历史单一参数组行为逐位一致（由 build_lora_param_groups 保证）。
    """
    value = float(train_cfg.get("lora_b_weight_decay", 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"training.lora_b_weight_decay 必须为非负有限值，实际 {value}"
        )
    return value


def apply_condition_dropout(
    cond: Optional[Tensor], p: float, generator: Optional[torch.Generator] = None
) -> Optional[Tensor]:
    """训练期 condition dropout：逐样本独立以概率 p 把整条 image_embeds 置零（纯函数）。

    与官方 CFG 空分支一致——用 ``zeros_like`` 置零（保持 dtype / 形状），
    让模型把「零嵌入 == 无条件」学进分布内；否则推理 CFG 的 uncond 分支
    对 LoRA 属分布外，高 cfg_scale 外推会放大损坏。

    Args:
        cond: 图像条件嵌入 [B, S, D]；None 或 p <= 0 时原样返回（零开销路径）。
        p: 逐样本置零概率（0~1）。
        generator: 可选随机数生成器（测试可复现用）；None 用全局 RNG。
            若非 None，其 device 必须与 ``cond.device`` 一致（torch.rand
            要求生成器与目标设备匹配，否则抛 RuntimeError）。

    Returns:
        置零后的条件张量（不修改输入本身）；无需 dropout 时返回原对象。
    """
    if cond is None or p <= 0.0:
        return cond
    batch_size = cond.shape[0]
    mask = torch.rand(batch_size, device=cond.device, generator=generator) < float(p)
    if not mask.any():
        return cond
    # 掩码广播到 [B, 1, ..., 1]：被选中的样本整条置零，其余逐元素不变
    shape = (batch_size,) + (1,) * (cond.dim() - 1)
    return torch.where(mask.view(*shape), torch.zeros_like(cond), cond)


def train_step(
    dit: nn.Module,
    latents: Tensor,
    use_bf16: bool,
    timestep_scale: float = 1000.0,
    cond: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
) -> Tensor:
    """单步扩散训练：采样 t -> 加噪 -> 预测速度 -> MSE。

    Rectified flow 约定（t ∈ [0, 1]，与 ``RectifiedFlowScheduler`` 一致）::

        x_t     = (1 - t) * x_0 + t * noise
        目标速度 v = noise - x_0   （即 dx_t / dt）

    Args:
        dit: 含 LoRA 的 DiT。
        latents: 干净 latent ``x_0``，[B, 2048, 64]。
        use_bf16: 是否启用 bf16 autocast。
        timestep_scale: 送入 DiT 的时间步缩放（真实调度器期望 [0, 1000]）。
        cond: DINOv2 图像条件嵌入 [B, S, 1024]；None 为无条件训练。
        sample_weights: [B] 逐样本损失权重（Learned Semantic Reward 训练加权）；
            None / 全 1 时退化为普通均值 MSE。

    Returns:
        标量 MSE 损失。
    """
    device = latents.device
    batch_size = latents.shape[0]

    noise = torch.randn_like(latents)
    t = torch.rand(batch_size, device=device, dtype=latents.dtype)  # U(0, 1)
    t_expand = t.view(batch_size, *([1] * (latents.dim() - 1)))

    x_t = (1.0 - t_expand) * latents + t_expand * noise
    velocity_target = noise - latents

    # 梯度检查点要求至少一个输入张量带梯度，x_t 作为锚点
    x_t = x_t.detach().requires_grad_(torch.is_grad_enabled())

    ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with ctx:
        velocity_pred = forward_dit(dit, x_t, t * timestep_scale, cond)
        if sample_weights is None:
            loss = F.mse_loss(velocity_pred.float(), velocity_target.float())
        else:
            # 逐样本 MSE -> 按 Critic 权重加权平均（权重均值已归一到 1.0，
            # 不改变损失的期望量级）
            per_sample = F.mse_loss(
                velocity_pred.float(), velocity_target.float(), reduction="none"
            )
            per_sample = per_sample.flatten(1).mean(dim=1)  # [B]
            loss = (per_sample * sample_weights.to(per_sample)).mean()
    return loss


@torch.no_grad()
def compute_validation_loss(
    dit: nn.Module,
    val_batches: List[Dict[str, Tensor]],
    use_bf16: bool,
    timestep_scale: float,
    device: torch.device,
    seed: int = VAL_NOISE_SEED,
) -> Optional[float]:
    """在 held-out 样本上计算验证 MSE（无梯度）。

    前向与 ``train_step`` 完全一致（rectified flow 加噪 + 速度场 MSE），但：

    1. 噪声与时间步由固定种子的 ``torch.Generator`` 产生，每次验证用同一批
       噪声，不同步数的验证损失可直接对比，同时不污染全局 RNG（训练可复现）；
    2. 不做 Critic 样本加权，始终是纯均值 MSE（作为干净的泛化指标）；
    3. 用当前训练权重（非 EMA），便于与 ``train/loss`` 直接对比判断过拟合。

    Args:
        dit: 含 LoRA 的 DiT（调用前后的 train / eval 模式会自动恢复）。
        val_batches: ``split_validation_batches`` 的输出；空列表直接返回 None。
        use_bf16: 是否启用 bf16 autocast（与训练一致）。
        timestep_scale: 时间步缩放（与训练一致）。
        device: 计算设备。
        seed: 噪声种子。

    Returns:
        平均验证 MSE；验证集为空或计算失败时返回 None（不中断训练）。
    """
    if not val_batches:
        return None

    was_training = dit.training
    dit.eval()
    total_loss = 0.0
    count = 0
    try:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        for batch in val_batches:
            latents = batch["latent"].to(device, non_blocking=True).float()
            cond = batch.get("image_embeds", None)
            if cond is not None:
                cond = cond.to(device, non_blocking=True).float()

            noise = torch.randn(
                latents.shape, generator=generator, device=device, dtype=latents.dtype
            )
            t = torch.rand(
                latents.shape[0], generator=generator, device=device, dtype=latents.dtype
            )
            t_expand = t.view(latents.shape[0], *([1] * (latents.dim() - 1)))
            x_t = (1.0 - t_expand) * latents + t_expand * noise
            velocity_target = noise - latents

            ctx = (
                torch.cuda.amp.autocast(dtype=torch.bfloat16)
                if use_bf16 and device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with ctx:
                velocity_pred = forward_dit(dit, x_t, t * timestep_scale, cond)
            # 按样本数加权：num_val_samples 不整除 val_batch_size 时尾部小
            # batch 不能与大 batch 等权，否则验证指标有偏
            total_loss += float(
                F.mse_loss(velocity_pred.float(), velocity_target.float()).item()
            ) * latents.shape[0]
            count += latents.shape[0]
    except Exception as exc:  # 验证只是观测指标，失败绝不能中断训练
        LOGGER.warning("[验证] 计算失败（%s），跳过本次验证", exc)
        return None
    finally:
        if was_training:
            dit.train()
    return total_loss / max(count, 1)


# ====================================================================== #
# 评估：Euler 采样 -> VAE 解码 -> mesh -> 多视角渲染
# ====================================================================== #
@torch.no_grad()
def sample_latent_euler(
    dit: nn.Module,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    num_steps: int = EVAL_NUM_STEPS,
    timestep_scale: float = 1000.0,
    cond: Optional[Tensor] = None,
    guidance_scale: float = EVAL_GUIDANCE_SCALE,
) -> Tensor:
    """用训练好的速度场从纯噪声做 Euler 反向积分，得到采样 latent。

    积分方向 t: 1 -> 0（x_1 = noise，x_0 = 数据），步长 dt = -1/num_steps。

    ``guidance_scale > 1.0`` 且 ``cond`` 非 None 时启用 classifier-free
    guidance（与 TripoSG 官方推理一致）：每步用零嵌入算无条件速度
    ``v_uncond``，用真实条件算 ``v_cond``，组合为
    ``v = v_uncond + guidance_scale * (v_cond - v_uncond)``。
    ``guidance_scale <= 1.0`` 或无条件（含全零条件回退）时退化为单分支采样，
    避免双倍前向开销。
    注意：CFG 只是评估/推理期采样技巧，不影响训练目标。
    """
    x = torch.randn((1, *latent_shape), device=device)
    dt = 1.0 / float(num_steps)
    # cond 全零时（无 image_embeds 的回退场景）v_cond == v_uncond，
    # CFG 双前向等价于无引导却多付一倍开销，直接退化为单分支
    use_cfg = cond is not None and guidance_scale > 1.0 and bool(cond.abs().any())
    cond_uncond = torch.zeros_like(cond) if use_cfg else None
    for step in range(num_steps):
        t = torch.full((1,), 1.0 - step * dt, device=device)
        if use_cfg:
            v_cond = forward_dit(dit, x, t * timestep_scale, cond)
            v_uncond = forward_dit(dit, x, t * timestep_scale, cond_uncond)
            velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            velocity = forward_dit(dit, x, t * timestep_scale, cond)
        x = x - dt * velocity
    return x.squeeze(0)


def _make_sdf_func(vae: Any, latent: Tensor):
    """构造 ``hierarchical_extract_geometry`` 需要的 SDF 查询函数。

    真实 ``TripoSGVAEModel.decode(z, sampled_points)`` 返回
    ``DecoderOutput(sample=SDF)``；包装成 points [1, N, 3] -> sdf [1, N, 1] 的可调用。

    注意：``hierarchical_extract_geometry`` 传入的 points 已带 batch 维
    （``xyz_samples.unsqueeze(0)``），这里绝不能再 unsqueeze，否则 4D 输入会触发
    attention processor 的 ndim==4 分支导致形状错乱。
    """
    latent_batch = latent.unsqueeze(0)  # [1, N_tok, 64]

    def sdf_func(points: Tensor) -> Tensor:
        output = vae.decode(latent_batch, sampled_points=points)
        return output.sample if hasattr(output, "sample") else output[0]

    return sdf_func


@torch.no_grad()
def decode_latent_to_mesh(
    vae: Any, latent: Tensor, octree_depth: int = EVAL_OCTREE_DEPTH
) -> Tuple[Tensor, Tensor]:
    """把 latent 解码为 SDF 并用分层八叉树提取归一化 mesh。

    调用 ``triposg.inference_utils.hierarchical_extract_geometry``
    （与 pipeline 的 mesh 提取路径完全一致）。

    Returns:
        (vertices [V, 3], faces [F, 3])，坐标在 [-1.005, 1.005]^3 域内。

    Raises:
        RuntimeError: 提取失败或结果为空。
    """
    from triposg.inference_utils import hierarchical_extract_geometry

    device = latent.device
    bounds = (-1.005,) * 3 + (1.005,) * 3
    results = hierarchical_extract_geometry(
        _make_sdf_func(vae, latent.to(device)),
        device=device,
        bounds=bounds,
        dense_octree_depth=max(octree_depth - 2, 4),
        hierarchical_octree_depth=octree_depth,
    )
    if not results:
        raise RuntimeError("八叉树提取未得到任何 mesh")
    verts, faces = results[0]
    # marching_cubes 失败时 triposg 返回 (None, None)，必须先兜住再转数组，
    # 否则 np.asarray(None) 抛 TypeError（不在打分侧常规捕获范围内）
    if verts is None or faces is None:
        raise RuntimeError("八叉树提取失败（marching cubes 返回空）")
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        raise RuntimeError("提取的 mesh 为空")

    # 归一化到单位球 + 外向 CCW 环绕（与 diffusion_adapter 的后处理一致）
    center = 0.5 * (verts.max(axis=0) + verts.min(axis=0))
    verts = verts - center
    radius = float(np.linalg.norm(verts, axis=1).max())
    if radius < 1e-8:
        raise RuntimeError("解码 mesh 退化")
    verts = verts / radius

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0
    outward = np.sum(face_normals * (centroids - verts.mean(axis=0, keepdims=True)), axis=1) > 0
    fixed = faces.copy()
    fixed[~outward] = faces[~outward][:, ::-1]

    return (
        torch.from_numpy(verts.astype(np.float32)),
        torch.from_numpy(fixed.astype(np.int64)),
    )


def _shard_caption(shard: Dict[str, Any], inner_idx: int) -> str:
    """读取 shard 内某条样本的 caption（缺失时返回空串）。"""
    captions = shard.get("captions", [])
    return str(captions[inner_idx]) if inner_idx < len(captions) else ""


def _sample_eval_conditions(
    dataset: LatentCacheDataset, num_samples: int, device: torch.device
) -> Tuple[List[Tensor], List[str]]:
    """从 latent cache 随机抽取评估用条件嵌入及其 caption。

    优先取 shard 中真实的 DINOv2 ``image_embeds``（本轮为条件训练，
    真实条件采样预览才有意义）；仅当整个 cache 都没有 image_embeds 时
    才回退零嵌入（与官方 CFG 空分支一致，避免 cross-attn 形状错）。

    Returns:
        ``(条件嵌入列表, caption 列表)``，两者长度一致且一一对应；caption
        供 CLIP Score 使用，缺失时为空串（对应样本不参与 CLIP 打分）。
    """
    conds: List[Tensor] = []
    captions: List[str] = []
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for idx in indices:
        if len(conds) >= num_samples:
            break
        shard_name, inner_idx = dataset.index[idx]
        shard = dataset._get_shard(shard_name)
        embeds = shard.get("image_embeds", None)
        if embeds is not None:
            conds.append(embeds[inner_idx].float().unsqueeze(0).to(device))
            captions.append(_shard_caption(shard, inner_idx))
    if conds:
        return conds, captions
    zero = torch.zeros((1, dataset.cond_num_tokens, dataset.cond_dim), device=device)
    captions = [
        _shard_caption(dataset._get_shard(dataset.index[idx][0]), dataset.index[idx][1])
        for idx in indices[:num_samples]
    ]
    return [zero.clone() for _ in range(num_samples)], captions


def _build_renderer_from_cfg(render_cfg: Dict[str, Any], device: torch.device) -> Any:
    """按 ``rendering`` 配置段构建多视角渲染器（缺失字段回落默认值）。

    与 ``inference.build_renderer`` 口径一致：``azimuth_strategy="fixed"``
    （等间距环绕视角，预览图与打分均可复现）。

    Args:
        render_cfg: ``rendering`` 配置段；None / 空 dict 时全部用默认值
            （image_size=256, num_views=4, camera_distance=2.5,
            elevation_range=[-30, 30]）。
        device: 渲染设备。

    Raises:
        Exception: nvdiffrast / CUDA 不可用时由 ``MultiViewRenderer`` 抛出，
            由调用方捕获并跳过评估。
    """
    from rendering.multi_view_render import MultiViewRenderer

    render_cfg = dict(render_cfg or {})
    elevation = list(render_cfg.get("elevation_range", (-30.0, 30.0)))
    if len(elevation) != 2:
        LOGGER.warning("[评估] rendering.elevation_range 应为 [min, max]，回落 [-30, 30]")
        elevation = [-30.0, 30.0]
    return MultiViewRenderer(
        image_size=int(render_cfg.get("image_size", 256)),
        num_views=int(render_cfg.get("num_views", 4)),
        camera_distance=float(render_cfg.get("camera_distance", 2.5)),
        elevation_range=(float(elevation[0]), float(elevation[1])),
        azimuth_strategy="fixed",
        device=str(device),
    )


def _build_eval_clip(config: Dict[str, Any], device: torch.device) -> Optional[Any]:
    """构建评估用的冻结 CLIP 编码器（进程内缓存），不可用时返回 None。

    ``evaluation.clip_model_name`` 为 null / 空串时视为主动关闭；导入或加载失败
    时只警告一次并缓存 None，之后每轮评估静默跳过 CLIP 相关指标。
    """
    if "clip" in _EVAL_MODEL_CACHE:
        return _EVAL_MODEL_CACHE["clip"]

    eval_cfg = dict(config.get("evaluation", {}) or {})
    model_name = eval_cfg.get("clip_model_name", "ViT-B/32")
    encoder: Optional[Any] = None
    if model_name:
        try:
            from vlm.clip_encoder import CLIPEncoder

            encoder = CLIPEncoder(
                model_name=str(model_name),
                device=str(device),
                input_range="zero_one",  # 渲染器输出已在 [0, 1]
                pretrained=str(eval_cfg.get("clip_pretrained", "openai")),
            )
            LOGGER.info("[评估] CLIP 已加载（%s），将记录 eval/clip_score", model_name)
        except Exception as exc:  # CLIP 为可选依赖，缺失不影响训练
            LOGGER.warning("[评估] CLIP 不可用（%s），跳过 clip_score / critic_score", exc)
            encoder = None
    _EVAL_MODEL_CACHE["clip"] = encoder
    return encoder


def _build_eval_critic(
    config: Dict[str, Any], device: torch.device
) -> Optional[Tuple[Any, int, Dict[str, Any]]]:
    """加载评估用的 SemanticCritic（进程内缓存），不可用时返回 None。

    权重取 ``evaluation.critic_checkpoint``（``train.py`` 产出的 GAN checkpoint，
    含 ``state['critic']``），复用 ``inference.build_critic`` 保证结构与训练一致。
    注意它与 ``training.critic_weights_path``（uid -> 权重 JSON）不是同一个文件。

    Returns:
        ``(critic, geo_dim, critic 训练时的 rendering 配置)``；
        未配置 / 加载失败时返回 None。
    """
    if "critic" in _EVAL_MODEL_CACHE:
        return _EVAL_MODEL_CACHE["critic"]

    checkpoint = (config.get("evaluation", {}) or {}).get("critic_checkpoint")
    bundle: Optional[Tuple[Any, int, Dict[str, Any]]] = None
    if checkpoint:
        try:
            from inference import build_critic

            critic, geo_dim, critic_render_cfg = build_critic(str(checkpoint), device)
            bundle = (critic, int(geo_dim), dict(critic_render_cfg or {}))
            LOGGER.info(
                "[评估] SemanticCritic 已加载（%s），将记录 eval/critic_score", checkpoint
            )
        except Exception as exc:  # Critic 为可选指标，加载失败不影响训练
            LOGGER.warning("[评估] Critic 不可用（%s），跳过 critic_score", exc)
            bundle = None
    _EVAL_MODEL_CACHE["critic"] = bundle
    return bundle


@torch.no_grad()
def _clip_score_for_views(clip_encoder: Any, views: Tensor, caption: str) -> float:
    """单个样本的 CLIP Score：逐视角图文余弦相似度的均值。

    口径与 ``evaluate.py:clip_score_batch`` 一致：原始余弦相似度（不做 ×100
    缩放），编码 ``augment=False`` 保证确定性。

    Args:
        clip_encoder: CLIPEncoder 实例。
        views: [N_views, 3, H, W] 渲染图，范围 [0, 1]。
        caption: 该样本的文本描述。

    Returns:
        均值余弦相似度。
    """
    image_features = clip_encoder.encode_images(views, augment=False)  # [N, D]
    text_features = clip_encoder.encode_text([caption])  # [1, D]
    text_features = text_features.expand(image_features.shape[0], -1)
    return float(clip_encoder.paired_similarity(image_features, text_features).mean().item())


@torch.no_grad()
def run_evaluation(
    iteration: int,
    dit: nn.Module,
    adapters: Dict[str, LoRALinear],
    ema: LoRAEMA,
    vae: Optional[Any],
    dataset: LatentCacheDataset,
    config: Dict[str, Any],
    device: torch.device,
    writer: Optional[Any],
) -> Optional[Dict[str, float]]:
    """定期评估：EMA 权重采样 latent -> 解码 mesh -> 渲染预览图 + 附加指标。

    渲染参数（image_size / num_views / camera_distance / elevation_range）全部
    从配置 ``rendering`` 段读取，缺失时回落原有默认值。预览图之外，在对应
    模块可用时额外写入两个指标：

    - ``eval/clip_score``：渲染图与样本 caption 的 CLIP 图文余弦相似度；
    - ``eval/critic_score``：SemanticCritic plausibility ∈ [0, 1]，打分渲染器
      按 Critic 训练时的 rendering 配置单独构建（与 ``inference.py`` 对齐）。

    评估（含两个附加指标）失败只记警告，绝不中断训练。

    Returns:
        本轮 eval 均值分数字典（键 ``clip_score`` / ``critic_score``，仅在
        对应指标有有效样本时存在）；评估未能进行（渲染器 / VAE 不可用）
        时返回 None。供 best checkpoint 跟踪计算复合分。
    """
    output_dir = config.get("logging", {}).get("output_dir", "./outputs/diffusion_train")
    eval_dir = os.path.join(output_dir, "eval")
    train_cfg = config.get("training", {})
    timestep_scale = float(train_cfg.get("timestep_scale", 1000.0))

    try:
        from utils.visualize import save_image, tile_images

        # 渲染参数一律读配置；旧配置无 rendering 段时等价于原来的硬编码默认值
        renderer = _build_renderer_from_cfg(config.get("rendering", {}), device)
    except Exception as exc:  # 渲染器依赖 nvdiffrast/CUDA，不可用时跳过
        LOGGER.warning("[评估] 渲染器构建失败（%s），跳过本次评估", exc)
        return

    # ---- 附加指标依赖的模型：任一不可用时为 None，对应指标静默跳过 ---- #
    clip_encoder = _build_eval_clip(config, device)
    # Critic 的语义输入来自 CLIP，CLIP 不可用时 Critic 打分也无法进行
    critic_bundle = _build_eval_critic(config, device) if clip_encoder is not None else None
    critic_renderer: Optional[Any] = None
    if critic_bundle is not None:
        try:
            # Critic 的预处理必须对齐其训练时的渲染配置，因此单独建渲染器
            critic_renderer = _build_renderer_from_cfg(critic_bundle[2], device)
        except Exception as exc:
            LOGGER.warning("[评估] Critic 打分渲染器构建失败（%s），跳过 critic_score", exc)
            critic_bundle = None

    # 用 EMA 权重评估（评估后恢复训练权重）
    backup = ema.apply(adapters)
    dit.eval()
    try:
        latent_shape = tuple(dataset.manifest.get("latent_shape") or ())
        if not latent_shape:
            shard = dataset._get_shard(dataset.index[0][0])
            latent_shape = tuple(shard["latents"].shape[1:])

        # 采样条件：优先从 cache 随机抽真实 DINOv2 image_embeds（本轮是
        # 条件训练，零条件采样只会出黑图）；cache 无 image_embeds 时才回退
        # 零嵌入（与官方 CFG 空分支一致）。不能传 None —— diffusers Attention
        # 在 encoder_hidden_states=None 时会拿 2048 维的 hidden_states 当
        # cross-attn KV，撞上 to_k/to_v 的 1024 维投影报形状错
        conds, captions = _sample_eval_conditions(dataset, EVAL_NUM_SAMPLES, device)

        # 评估前清理显存缓存：frozen DiT + VAE + AdamW 状态仍常驻显存，
        # 高八叉树深度解码 + CFG 双前向会抬高峰值，提前清缓存降低 OOM 概率
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if vae is None:
            LOGGER.warning("[评估] VAE 不可用，跳过 mesh 解码预览")
            return

        tiles = []
        clip_scores: List[float] = []
        critic_scores: List[float] = []
        for i in range(EVAL_NUM_SAMPLES):
            try:
                latent = sample_latent_euler(
                    dit,
                    latent_shape,
                    device,
                    EVAL_NUM_STEPS,
                    timestep_scale,
                    conds[i % len(conds)],
                    guidance_scale=EVAL_GUIDANCE_SCALE,
                )
                vertices, faces = decode_latent_to_mesh(vae, latent)
                verts = vertices.unsqueeze(0).to(device)
                tris = faces.to(device)
                out = renderer.render(verts, tris)
                views = out["images"][0]  # [N_views, 3, H, W]
                tiles.append(views.cpu())
            except Exception as exc:
                LOGGER.warning("[评估] 样本 %d 生成失败（%s），跳过", i, exc)
                continue

            # ---- 附加指标：单个样本打分失败不影响其他样本与预览图 ---- #
            caption = captions[i % len(captions)] if captions else ""
            if clip_encoder is not None and caption:
                try:
                    clip_scores.append(_clip_score_for_views(clip_encoder, views, caption))
                except Exception as exc:
                    LOGGER.warning("[评估] 样本 %d CLIP 打分失败（%s），跳过", i, exc)
            if critic_bundle is not None and critic_renderer is not None:
                try:
                    from inference import critic_score_for_mesh

                    critic_scores.append(
                        critic_score_for_mesh(
                            critic_bundle[0],
                            critic_bundle[1],
                            clip_encoder,
                            critic_renderer,
                            verts,
                            tris,
                        )
                    )
                except Exception as exc:
                    LOGGER.warning("[评估] 样本 %d Critic 打分失败（%s），跳过", i, exc)

        if tiles:
            grid = tile_images(torch.cat(tiles, dim=0), ncols=8)
            image_path = os.path.join(eval_dir, f"sample_{iteration:07d}.png")
            save_image(grid, image_path)
            LOGGER.info("[评估] 已保存 %d 个采样预览 -> %s", len(tiles), image_path)
            if writer is not None:
                grid_np = np.asarray(grid)
                if grid_np.ndim == 2:
                    grid_np = np.stack([grid_np] * 3, axis=-1)
                writer.add_image("eval/samples", grid_np, iteration, dataformats="HWC")

        # ---- 附加指标汇总（无有效样本的指标不写日志）---- #
        scores: Dict[str, float] = {}
        for key, values in (("clip_score", clip_scores), ("critic_score", critic_scores)):
            if not values:
                continue
            mean_value = sum(values) / len(values)
            scores[key] = mean_value
            LOGGER.info("[评估] %s %.4f（%d 个样本）", key, mean_value, len(values))
            if writer is not None:
                writer.add_scalar(f"eval/{key}", mean_value, iteration)
        return scores
    finally:
        LoRAEMA.restore(adapters, backup)
        dit.train()


# ====================================================================== #
# Checkpoint
# ====================================================================== #
def save_checkpoint(
    path: str,
    adapters: Dict[str, LoRALinear],
    ema: LoRAEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    iteration: int,
    config: Dict[str, Any],
) -> None:
    """保存 LoRA 权重、EMA、优化器 / 调度器状态与迭代数。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state: Dict[str, Any] = {
        "iteration": iteration,
        "lora": lora_state_dict(adapters),
        "ema": {k: v.clone() for k, v in ema.shadow.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
    }
    torch.save(state, path)
    LOGGER.info("已保存 checkpoint: %s", path)


def load_checkpoint(
    path: str,
    adapters: Dict[str, LoRALinear],
    ema: LoRAEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> int:
    """从 checkpoint 恢复训练，返回起始迭代数。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    state = torch.load(path, map_location=device)

    load_lora_state_dict(adapters, state["lora"])
    if "ema" in state:
        for key, value in ema.shadow.items():
            if key in state["ema"]:
                value.copy_(state["ema"][key].to(value.device))
    else:
        LOGGER.warning("checkpoint 缺少 EMA 状态，EMA 将以当前 LoRA 权重重新起步")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    start_iteration = int(state.get("iteration", 0)) + 1
    LOGGER.info("已从 %s 恢复，起始迭代 %d", path, start_iteration)
    return start_iteration


class BestCheckpointer:
    """按生成质量跟踪 best checkpoint（复合分 = eval clip × critic）。

    复合分取训练内 eval 的 ``clip_score`` 均值（CLIP 图文余弦相似度，
    有界于 [-1, 1]）与 ``critic_score`` 均值（SemanticCritic plausibility，
    有界于 [0, 1]）的**原始乘积**，不做归一化：两项指标本身有界且方向一致
    （越大越好），乘积对二者均单调，任一指标退化都会拉低复合分；归一化
    反而会引入对历史分数序列的依赖，不可复现。

    仅当两项指标都有效时才参与 best 竞争；复合分严格变高才刷新。
    """

    def __init__(self) -> None:
        self.best_composite: Optional[float] = None
        self.best_iteration: Optional[int] = None

    def offer(self, iteration: int, clip_score: float, critic_score: float) -> bool:
        """提交一次 eval 分数；复合分严格更高时刷新 best 并返回 True。

        复合分为 NaN / inf（任一指标非有限）时打 warning 并直接 return False，
        不参与 best 竞争，防止脏分数污染 best 记录。
        """
        composite = float(clip_score) * float(critic_score)
        if not math.isfinite(composite):
            LOGGER.warning(
                "[best] eval 分数含非有限值（clip=%s, critic=%s），跳过本轮 best 竞争",
                clip_score,
                critic_score,
            )
            return False
        if self.best_composite is None or composite > self.best_composite:
            self.best_composite = composite
            self.best_iteration = int(iteration)
            return True
        return False


# ====================================================================== #
# 主流程
# ====================================================================== #
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="TripoSG LoRA diffusion fine-tuning")
    parser.add_argument("--config", type=str, default="configs/train_diffusion.yaml")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})

    output_dir = str(log_cfg.get("output_dir", "./outputs/diffusion_train"))
    setup_logging(output_dir)
    set_seed(args.seed)

    log_interval = int(log_cfg.get("log_interval", 50))
    save_interval = int(log_cfg.get("save_interval", 2000))
    eval_interval = int(log_cfg.get("eval_interval", 1000))
    max_iterations = int(train_cfg.get("max_iterations", 50000))
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    use_bf16 = bool(train_cfg.get("use_bf16", True))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    timestep_scale = float(train_cfg.get("timestep_scale", 1000.0))
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    # Condition dropout：训练期逐样本以概率 p 把 image_embeds 置零，让 CFG 的
    # uncond 零嵌入分支留在分布内；0.0 = 不启用，训练行为与原来完全一致。
    # 区间校验（含 NaN）在 parse_condition_dropout 内完成
    condition_dropout = parse_condition_dropout(train_cfg)

    # 附加评估配置（旧配置缺失 evaluation 段时全部回落默认值；
    # num_val_samples 默认 0 = 不切验证集，训练行为与原来完全一致）
    eval_cfg = config.get("evaluation", {}) or {}
    val_interval = int(eval_cfg.get("interval", 500))
    num_val_samples = int(eval_cfg.get("num_val_samples", 0))
    val_batch_size = int(eval_cfg.get("val_batch_size", 4))

    device = torch.device(args.device)
    LOGGER.info("扩散 LoRA 微调启动：device=%s, bf16=%s", device, use_bf16)
    LOGGER.info(
        "condition_dropout=%s",
        condition_dropout if condition_dropout > 0.0 else "0.0（未启用）",
    )

    # ---- 模型 / 优化器 / 调度器 ----
    dit, adapters, vae = build_dit_with_lora(config, device)
    if bool(train_cfg.get("gradient_checkpointing", True)):
        num_wrapped = enable_gradient_checkpointing(dit)
        LOGGER.info("梯度检查点：包装了 %d 个 transformer block", num_wrapped)

    lora_params = [
        param for adapter in adapters.values() for param in (adapter.lora_A, adapter.lora_B)
    ]
    # B 矩阵锚：lora_B 单独一组施加 training.lora_b_weight_decay（缺省 0.0 时
    # 退化为原有单一参数组，训练行为逐位不变）；梯度裁剪仍对全量 LoRA 参数
    lora_b_weight_decay = parse_lora_b_weight_decay(train_cfg)
    if lora_b_weight_decay > 0.0:
        LOGGER.info("B 矩阵锚已启用：lora_B weight_decay=%.4g", lora_b_weight_decay)
    param_groups = build_lora_param_groups(adapters, lora_b_weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=learning_rate, weight_decay=DEFAULT_WEIGHT_DECAY
    )
    # 调度器按实际优化器步数构建（每 accumulation_steps 个迭代才 step 一次），
    # 与训练循环中 scheduler.step() 的调用频率保持一致
    total_optim_steps = max(1, max_iterations // accumulation_steps)
    scheduler = build_scheduler(optimizer, warmup_steps, total_optim_steps)
    ema = LoRAEMA(adapters, ema_decay)

    # ---- 数据 ----
    cache_dir = str(config.get("data", {}).get("latent_cache_dir", "./cache/triposg_latents"))
    critic_weights_path = train_cfg.get("critic_weights_path", None)
    dataset = LatentCacheDataset(
        cache_dir,
        critic_weights_path=str(critic_weights_path) if critic_weights_path else None,
    )
    # 样本数不足 batch_size 时 drop_last=True 会让 infinite_batches 永远不产出，提前报错
    batch_size = int(train_cfg.get("batch_size", 32))
    if len(dataset) < batch_size:
        raise RuntimeError(
            f"数据集样本数 ({len(dataset)}) 小于 batch_size ({batch_size})，"
            f"请减小 batch_size 或增大缓存数据量"
        )

    # ---- 可选 held-out 验证集（必须在建 DataLoader 之前切分，否则 worker
    # 拿到的是切分前的索引）；num_val_samples=0 时训练集与原来完全一致 ----
    val_batches = split_validation_batches(
        dataset, num_val_samples, val_batch_size, min_train_samples=batch_size
    )
    if val_batches:
        num_held_out = sum(item["latent"].shape[0] for item in val_batches)
        LOGGER.info(
            "验证集：%d 个 held-out 样本（%d 个 batch），每 %d 步计算一次 MSE",
            num_held_out,
            len(val_batches),
            val_interval,
        )
    elif num_val_samples > 0:
        LOGGER.warning(
            "样本数不足以切出 %d 个验证样本（需留至少 %d 个给训练），已跳过验证集切分",
            num_val_samples,
            batch_size,
        )

    loader = build_dataloader(config, dataset)
    batches = infinite_batches(loader)
    LOGGER.info("latent 数据集：%d 个训练样本（%s）", len(dataset), cache_dir)

    # ---- Learned Semantic Reward 训练加权统计 ----
    if dataset.critic_weights:
        weighted = sum(
            1 for shard_name, inner_idx in dataset.index
            if _dataset_uid_weighted(dataset, shard_name, inner_idx)
        )
        # 覆盖率扫描会把全部 shard 读进进程缓存，统计完立即释放内存
        dataset._shard_cache.clear()
        values = list(dataset.critic_weights.values())
        LOGGER.info(
            "Critic 训练加权已启用：%s（权重表 %d 条，样本覆盖率 %.1f%%，"
            "weight 均值 %.3f min %.3f max %.3f；缺失 uid 按 1.0）",
            critic_weights_path,
            len(values),
            100.0 * weighted / max(len(dataset), 1),
            sum(values) / len(values),
            min(values),
            max(values),
        )

    # ---- 断点续训 ----
    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(
            args.resume, adapters, ema, optimizer, scheduler, device
        )

    # ---- TensorBoard ----
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(os.path.join(output_dir, "tb"))
    else:
        LOGGER.warning("tensorboard 未安装，跳过标量 / 图像日志")

    # ---- 训练循环（训练步内不做任何渲染 / VAE 调用）----
    optimizer.zero_grad(set_to_none=True)
    # best checkpoint 跟踪（配方 4）：training.save_best_checkpoint 显式开启才生效，
    # 旧配置缺省该字段时不产生任何新行为；复合分 = eval clip × critic
    # （未归一化，见 BestCheckpointer）；断点续训时不恢复历史 best，
    # 由续训后首轮达标 eval 重新竞争
    save_best_checkpoint = bool(train_cfg.get("save_best_checkpoint", False))
    best_checkpointer = BestCheckpointer()
    if save_best_checkpoint:
        LOGGER.info("best checkpoint 跟踪已启用：复合分 = eval clip × critic（未归一化）")
    loss_window: List[float] = []
    weight_window: List[float] = []
    grad_norm = 0.0
    time_start = time.time()

    for iteration in range(start_iteration, max_iterations):
        batch = next(batches)
        latents = batch["latent"].to(device, non_blocking=True)
        cond = batch.get("image_embeds", None)
        if cond is not None:
            cond = cond.to(device, non_blocking=True)
        # Condition dropout 仅训练分支生效（验证前向 compute_validation_loss
        # 不经过此处）；p=0 时 if 门控短路，行为逐条不变
        if cond is not None and condition_dropout > 0.0:
            cond = apply_condition_dropout(cond, condition_dropout)
        # Learned Semantic Reward 训练加权：未配置 critic_weights_path 时恒为 None，行为不变
        sample_weights = None
        if dataset.critic_weights:
            sample_weights = batch["weight"].to(device, non_blocking=True)

        loss = train_step(
            dit, latents, use_bf16, timestep_scale, cond, sample_weights
        ) / accumulation_steps
        loss.backward()

        loss_window.append(loss.item() * accumulation_steps)
        if sample_weights is not None:
            weight_window.append(float(sample_weights.mean().item()))

        if (iteration + 1) % accumulation_steps == 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP).item()
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(adapters)

        # ---- 日志（文本日志不依赖 TensorBoard）----
        if iteration % log_interval == 0 and loss_window:
            avg_loss = sum(loss_window) / len(loss_window)
            current_lr = scheduler.get_last_lr()[0]
            weight_info = ""
            if weight_window:
                avg_weight = sum(weight_window) / len(weight_window)
                weight_info = f" | sample_weight {avg_weight:.3f}"
            LOGGER.info(
                "iter %d/%d | loss %.4f | lr %.2e | grad_norm %.3f%s",
                iteration,
                max_iterations,
                avg_loss,
                current_lr,
                grad_norm,
                weight_info,
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, iteration)
                writer.add_scalar("train/lr", current_lr, iteration)
                writer.add_scalar("train/grad_norm", grad_norm, iteration)
                if weight_window:
                    writer.add_scalar(
                        "train/sample_weight", sum(weight_window) / len(weight_window), iteration
                    )
            loss_window.clear()
            weight_window.clear()

        # ---- 定期验证损失（held-out MSE，与 train/loss 对照可观察过拟合）----
        if val_batches and val_interval > 0 and (iteration + 1) % val_interval == 0:
            val_loss = compute_validation_loss(
                dit, val_batches, use_bf16, timestep_scale, device
            )
            if val_loss is not None:
                LOGGER.info("iter %d | val_loss %.4f", iteration + 1, val_loss)
                if writer is not None:
                    writer.add_scalar("eval/val_loss", val_loss, iteration + 1)

        # ---- 定期评估 ----
        if eval_interval > 0 and (iteration + 1) % eval_interval == 0:
            # 评估是观测手段而非训练关键路径：内部仅 try/finally，异常会
            # 逃逸出来直接中断训练，这里兜底捕获（不吞 KeyboardInterrupt）
            eval_scores: Optional[Dict[str, float]] = None
            try:
                eval_scores = run_evaluation(
                    iteration + 1, dit, adapters, ema, vae, dataset, config, device, writer
                )
            except Exception as exc:
                LOGGER.warning("[评估] 本轮评估整体失败，已跳过，训练继续：%s", exc)

            # ---- 按生成质量留 best checkpoint（复合分 = clip × critic）----
            if (
                save_best_checkpoint
                and eval_scores
                and "clip_score" in eval_scores
                and "critic_score" in eval_scores
                and best_checkpointer.offer(
                    iteration + 1, eval_scores["clip_score"], eval_scores["critic_score"]
                )
            ):
                LOGGER.info(
                    "BEST_UPDATE | iter %d | clip %.4f | critic %.4f | composite %.6f",
                    iteration + 1,
                    eval_scores["clip_score"],
                    eval_scores["critic_score"],
                    best_checkpointer.best_composite,
                )
                if writer is not None:
                    writer.add_scalar(
                        "eval/best_composite", best_checkpointer.best_composite, iteration + 1
                    )
                # 与 ckpt_final.pt 相同的 save_checkpoint 结构（含 lora / ema）；
                # 先写 .tmp 再 os.replace 原子替换，避免写一半被打断留下
                # 损坏的 ckpt_best.pt；保存失败降级为 warning 不中断训练
                # （与 eval 兜底策略一致）
                best_path = os.path.join(output_dir, "ckpt_best.pt")
                best_tmp_path = best_path + ".tmp"
                try:
                    save_checkpoint(
                        best_tmp_path, adapters, ema, optimizer, scheduler, iteration + 1, config
                    )
                    os.replace(best_tmp_path, best_path)
                except Exception as exc:
                    LOGGER.warning(
                        "[best] ckpt_best.pt 保存失败，降级跳过（训练继续）：%s", exc
                    )
                    if os.path.isfile(best_tmp_path):
                        try:
                            os.remove(best_tmp_path)
                        except OSError:
                            pass

        # ---- 定期保存 ----
        if save_interval > 0 and (iteration + 1) % save_interval == 0:
            save_checkpoint(
                os.path.join(output_dir, f"ckpt_{iteration + 1:07d}.pt"),
                adapters,
                ema,
                optimizer,
                scheduler,
                iteration + 1,
                config,
            )

    # ---- 收尾：最终 checkpoint + 导出纯 LoRA 权重（便于推理侧加载）----
    if save_best_checkpoint and best_checkpointer.best_composite is None:
        # best 从未产出：说明整轮没有一次 eval 同时拿到有效 clip 与 critic 分数
        LOGGER.warning(
            "save_best_checkpoint=true 但整轮训练未产出 ckpt_best.pt："
            "从未出现 clip_score 与 critic_score 同时有效的评估。请检查 "
            "evaluation.clip_model_name / evaluation.critic_checkpoint 是否可用、"
            "VAE 解码与渲染链路是否正常（eval 日志中的警告可定位缺失环节）"
        )
    final_path = os.path.join(output_dir, "ckpt_final.pt")
    save_checkpoint(final_path, adapters, ema, optimizer, scheduler, max_iterations, config)
    torch.save(
        {"lora": lora_state_dict(adapters), "ema": dict(ema.shadow)},
        os.path.join(output_dir, "lora_weights.pt"),
    )
    if writer is not None:
        writer.close()

    elapsed = time.time() - time_start
    LOGGER.info(
        "训练完成：%d 步，耗时 %.1f 分钟，输出目录 %s",
        max_iterations - start_iteration,
        elapsed / 60.0,
        output_dir,
    )


if __name__ == "__main__":
    main()

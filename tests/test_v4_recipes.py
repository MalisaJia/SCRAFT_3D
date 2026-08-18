"""v4 训练配方轻量单元测试（无 GPU，mock DiT 树）。

覆盖：
a) 新旧 target 匹配逻辑的挂载点数量（84 vs 42）与推理侧复现；
b) 参数组拆分：lora_B 带独立 weight_decay、lora_A 保持默认；缺省 0.0 时
   与旧行为（扁平参数列表）逐位一致；
c) BestCheckpointer 在模拟分数序列下的刷新逻辑（含 NaN 防护）；
d) load_lora_state_dict 未消费键校验；
e) LatentCacheDataset 混装缓存护栏；
f) lora_b_weight_decay 配置校验；
g) eval 采样器（推理同款 rectified-flow）：eval_num_steps 校验、sigma 网格、
   积分方向、CFG 组合方式与噪声可复现性。
h) 训练目标速度场符号（v1.5 修正）：``v = x₀ - ε`` 与基座
   ``RectifiedFlowScheduler.step`` 的积分恒等式自洽，且 train_step 实际用的就是该符号。

运行：python tests/test_v4_recipes.py
"""

import os
import sys
import tempfile
import types

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# yaml 仅为 train_diffusion.load_config 所需，本测试不用；环境缺失时打桩
try:
    import yaml  # noqa: F401
except ImportError:
    stub = types.ModuleType("yaml")
    stub.safe_load = lambda *_a, **_k: {}
    sys.modules["yaml"] = stub

import torch
import torch.nn as nn

from train_diffusion import (  # noqa: E402
    DEFAULT_WEIGHT_DECAY,
    EVAL_FLOW_SHIFT,
    EVAL_GUIDANCE_SCALE,
    EVAL_NUM_STEPS,
    BestCheckpointer,
    LatentCacheDataset,
    _EVAL_MODEL_CACHE,
    _match_target,
    build_lora_param_groups,
    flow_match_sigmas,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    parse_eval_num_steps,
    parse_lora_b_weight_decay,
    resolve_flow_shift,
    sample_latent_flow,
    train_step,
)

RESULTS = []


def check(name: str, condition: bool) -> None:
    RESULTS.append((name, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")


# ------------------------------------------------------------------ #
# mock DiT：blocks.{i}.attn1/attn2.{to_q,to_k,to_v,to_out}
# ------------------------------------------------------------------ #
class MockAttention(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)


class MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = MockAttention()  # self-attention
        self.attn2 = MockAttention()  # cross-attention


class MockDiT(nn.Module):
    def __init__(self, num_blocks: int = 21):
        super().__init__()
        self.blocks = nn.ModuleList(MockBlock() for _ in range(num_blocks))


def test_target_matching() -> None:
    print("[a] target 匹配与挂载点数量")
    old_targets = ["to_q", "to_v"]
    new_targets = ["attn2.to_q", "attn2.to_v"]

    # 裸名：叶子名子串匹配，21 层 × 2 注意力 × 2 目标 = 84
    adapters_old = inject_lora(MockDiT(21), rank=4, target_modules=list(old_targets))
    check("旧式裸名 targets 挂载 84 个适配器", len(adapters_old) == 84)
    check(
        "旧式挂载覆盖 attn1 与 attn2",
        any(".attn1." in k for k in adapters_old) and any(".attn2." in k for k in adapters_old),
    )

    # 带点后缀模式：仅 cross-attention，21 层 × 2 目标 = 42
    adapters_new = inject_lora(MockDiT(21), rank=4, target_modules=list(new_targets))
    check("带点模式 targets 挂载 42 个适配器", len(adapters_new) == 42)
    check("带点模式全部命中 attn2（无 attn1）", all(".attn2." in k for k in adapters_new))
    check(
        "带点模式只含 to_q/to_v",
        all(k.endswith("attn2.to_q") or k.endswith("attn2.to_v") for k in adapters_new),
    )

    # 匹配函数本身的边界用例
    check("后缀匹配命中深层全名", _match_target("blocks.3.attn2.to_q", ["attn2.to_q"]))
    check("后缀匹配不误伤 attn1", not _match_target("blocks.3.attn1.to_q", ["attn2.to_q"]))
    check("带点模式不做子串误配", not _match_target("blocks.3.xattn2.to_q", ["attn2.to_q"]))
    check("裸名保持叶子子串匹配", _match_target("blocks.3.attn1.to_q", ["to_q"]))
    check("裸名别名展开仍有效", _match_target("blocks.3.attn1.to_q", ["q_proj"]))

    # 推理侧兼容：同一 targets 列表在另一棵同构树上复现 42 个挂点，
    # 且训练侧导出的权重可无损写回（lora_state_dict / load_lora_state_dict）
    adapters_infer = inject_lora(MockDiT(21), rank=4, target_modules=list(new_targets))
    check("推理侧同 targets 复现 42 个挂点", set(adapters_infer) == set(adapters_new))
    state = lora_state_dict(adapters_new)
    load_lora_state_dict(adapters_infer, state)
    same = all(
        torch.equal(adapters_infer[k].lora_A, adapters_new[k].lora_A)
        and torch.equal(adapters_infer[k].lora_B, adapters_new[k].lora_B)
        for k in adapters_new
    )
    check("训练权重可写回推理侧同构挂点树", same)


def test_param_groups() -> None:
    print("[b] 参数组拆分（B 矩阵锚）")
    adapters = inject_lora(MockDiT(2), rank=4, target_modules=["attn2.to_q", "attn2.to_v"])
    n = len(adapters)

    # 缺省 0.0：与旧实现一致的扁平列表（逐适配器 A/B 交错）
    flat = build_lora_param_groups(adapters, 0.0)
    legacy_flat = [
        param for adapter in adapters.values() for param in (adapter.lora_A, adapter.lora_B)
    ]
    check("缺省 0.0 返回扁平列表且与旧顺序一致", flat == legacy_flat)

    # 启用：两组，B 组带 lora_b_weight_decay，A 组保持默认
    groups = build_lora_param_groups(adapters, 1e-2)
    check("启用时返回 2 个参数组", isinstance(groups, list) and len(groups) == 2)
    b_ids = {id(a.lora_B) for a in adapters.values()}
    a_ids = {id(a.lora_A) for a in adapters.values()}
    check("lora_B 组含全部 B 且 weight_decay=1e-2",
          groups[0]["weight_decay"] == 1e-2 and {id(p) for p in groups[0]["params"]} == b_ids)
    check("lora_A 组含全部 A 且保持默认 weight_decay",
          groups[1]["weight_decay"] == DEFAULT_WEIGHT_DECAY
          and {id(p) for p in groups[1]["params"]} == a_ids)

    # 真实优化器验收：AdamW 每组生效的 weight_decay
    optimizer = torch.optim.AdamW(groups, lr=3e-5, weight_decay=DEFAULT_WEIGHT_DECAY)
    wd = [g["weight_decay"] for g in optimizer.param_groups]
    check("AdamW 组内 weight_decay = [1e-2, 0.01]", wd == [1e-2, DEFAULT_WEIGHT_DECAY])
    total = sum(len(g["params"]) for g in optimizer.param_groups)
    check("优化器参数总数 = 2 × 适配器数", total == 2 * n)

    # 缺省路径下优化器结构与历史单一默认组一致
    optimizer_legacy = torch.optim.AdamW(flat, lr=3e-5, weight_decay=DEFAULT_WEIGHT_DECAY)
    check("缺省路径优化器仍为单一默认参数组",
          len(optimizer_legacy.param_groups) == 1
          and optimizer_legacy.param_groups[0]["weight_decay"] == DEFAULT_WEIGHT_DECAY)


def test_best_tracking() -> None:
    print("[c] BestCheckpointer 复合分刷新逻辑")
    tracker = BestCheckpointer()
    # (iter, clip, critic, 是否应刷新)：复合分 = clip × critic（未归一化）
    sequence = [
        (100, 0.20, 0.50, True),   # 首次：0.100 建立基线
        (200, 0.19, 0.90, True),   # 0.171 > 0.100
        (300, 0.30, 0.40, False),  # 0.120 < 0.171（clip 升但 critic 掉）
        (400, 0.19, 0.90, False),  # 等于 best，不刷新（需严格更高）
        (500, 0.25, 0.80, True),   # 0.200 新 best
        (600, 0.05, 0.99, False),  # 0.0495 退化
        (700, float("nan"), 0.9, False),  # 非有限分数不参与竞争
    ]
    expected_composites = {100: 0.20 * 0.50, 200: 0.19 * 0.90, 500: 0.25 * 0.80}
    updates = []
    for iteration, clip, critic, should_update in sequence:
        got = tracker.offer(iteration, clip, critic)
        check(f"iter {iteration} 刷新判定 = {should_update}", got == should_update)
        if got:
            updates.append(iteration)

    check("共刷新 3 次且 iter 序列正确", updates == [100, 200, 500])
    check("best_iteration 指向最后一次刷新（NaN 未污染）", tracker.best_iteration == 500)
    check(
        "best_composite = 0.25 × 0.80",
        abs(tracker.best_composite - expected_composites[500]) < 1e-12,
    )


def _make_cache_dir(tmp_dir: str, token_sizes) -> str:
    """造一个含 num_shards 个 shard 的 latent 缓存（token_sizes 逐项指定
    各 shard 的 image_embeds token 数；None 表示该 shard 不含 image_embeds）。"""
    import json as _json

    shards = []
    for i, tokens in enumerate(token_sizes):
        shard_name = f"latent_shard_{i}.pt"
        shard = {
            "latents": torch.randn(2, 4, 64),
            "captions": ["a chair", "a chair"],
            "uids": [f"uid_{i}_0", f"uid_{i}_1"],
        }
        if tokens is not None:
            shard["image_embeds"] = torch.randn(2, int(tokens), 8)
        torch.save(shard, os.path.join(tmp_dir, shard_name))
        shards.append({"path": shard_name, "num_samples": 2})
    with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        _json.dump({"shards": shards}, handle)
    return tmp_dir


def test_unconsumed_keys() -> None:
    print("[d] load_lora_state_dict 未消费键校验")
    adapters_42 = inject_lora(MockDiT(2), rank=4, target_modules=["attn2.to_q", "attn2.to_v"])
    adapters_84 = inject_lora(MockDiT(2), rank=4, target_modules=["to_q", "to_v"])
    # 一致挂点树：正常加载（无额外键）
    state = lora_state_dict(adapters_42)
    load_lora_state_dict(adapters_42, state)
    check("键全匹配时正常加载", True)
    # 84 挂点权重加载到 42 挂点树：attn1 键未消费，必须 KeyError
    state_84 = lora_state_dict(adapters_84)
    try:
        load_lora_state_dict(adapters_42, state_84)
        check("未消费键拦下 84→42 静默丢权重", False)
    except KeyError as exc:
        check("未消费键拦下 84→42 静默丢权重", "未被任何已挂载适配器消费" in str(exc))


def test_mixed_cache_guard() -> None:
    print("[e] LatentCacheDataset 混装缓存护栏")
    # 混装：1370 与 257 同缓存 -> 初始化即 RuntimeError
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_cache_dir(tmp_dir, [1370, 257])
        try:
            LatentCacheDataset(tmp_dir)
            check("混装缓存被 RuntimeError 拦截", False)
        except RuntimeError as exc:
            msg = str(exc)
            check(
                "混装缓存被 RuntimeError 拦截且报出冲突 shard / 形状",
                "混装" in msg and "latent_shard_0.pt" in msg and "latent_shard_1.pt" in msg,
            )
    # 一致 257：正常构造且 token 维探测正确
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_cache_dir(tmp_dir, [257, 257])
        dataset = LatentCacheDataset(tmp_dir)
        check("一致 257 缓存正常构造", dataset.cond_num_tokens == 257 and dataset.cond_dim == 8)
    # 全无 image_embeds：回落默认形状
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_cache_dir(tmp_dir, [None, None])
        dataset = LatentCacheDataset(tmp_dir)
        check(
            "无 image_embeds 缓存回落默认 257×1024",
            dataset.cond_num_tokens == 257 and dataset.cond_dim == 1024,
        )


def test_lora_b_decay_validation() -> None:
    print("[f] lora_b_weight_decay 配置校验")
    check("缺省回落 0.0", parse_lora_b_weight_decay({}) == 0.0)
    check("合法正值通过", parse_lora_b_weight_decay({"lora_b_weight_decay": 1e-2}) == 1e-2)
    for bad in (-0.1, float("nan"), float("inf")):
        try:
            parse_lora_b_weight_decay({"lora_b_weight_decay": bad})
            check(f"非法值 {bad} 被 ValueError 拦截", False)
        except ValueError:
            check(f"非法值 {bad} 被 ValueError 拦截", True)


# ------------------------------------------------------------------ #
# mock DiT：速度场可解析，用于验证采样循环的积分方向与 CFG 组合
# ------------------------------------------------------------------ #
class MockFlowDiT(nn.Module):
    """速度场 = 每样本条件嵌入绝对值均值（无条件行 -> 0），并记录调用轨迹。

    这样 ``sample_latent_flow`` 的输出可闭式推导：单分支时
    ``x_T + sum(sigma_i - sigma_{i+1}) * v = x_T + (sigmas[0] - sigmas[-1]) * v``。
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.batch_sizes = []
        self.timesteps = []
        # 需要至少一个参数，便于外部按常规模型对待
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, timestep, encoder_hidden_states=None, return_dict=True):
        self.batch_sizes.append(int(hidden_states.shape[0]))
        self.timesteps.append(float(timestep.reshape(-1)[0].item()))
        if encoder_hidden_states is None:
            scale = torch.zeros(hidden_states.shape[0], device=hidden_states.device)
        else:
            scale = encoder_hidden_states.abs().flatten(1).mean(dim=1)
        velocity = torch.ones_like(hidden_states) * scale.view(-1, 1, 1)
        return (velocity,) if not return_dict else velocity


def test_eval_sampler() -> None:
    print("[g] eval 采样器（推理同款 rectified-flow）")
    device = torch.device("cpu")

    # --- eval_num_steps 配置校验：缺省 = 推理侧 50 ---
    check("eval_num_steps 缺省 50（与推理一致）",
          parse_eval_num_steps({}) == EVAL_NUM_STEPS == 50)
    check("eval_num_steps 显式值生效", parse_eval_num_steps({"eval_num_steps": 30}) == 30)
    for bad in (0, -5):
        try:
            parse_eval_num_steps({"eval_num_steps": bad})
            check(f"非法 eval_num_steps {bad} 被 ValueError 拦截", False)
        except ValueError:
            check(f"非法 eval_num_steps {bad} 被 ValueError 拦截", True)

    # --- sigma 网格：官方 set_timesteps 的形状与端点 ---
    sigmas = flow_match_sigmas(50, EVAL_FLOW_SHIFT)
    check("sigmas 长度 = N + 1", tuple(sigmas.shape) == (51,))
    check("首个 sigma = 1（纯噪声）", abs(float(sigmas[0]) - 1.0) < 1e-6)
    check("末个 sigma = 0（干净 latent）", abs(float(sigmas[-1])) < 1e-12)
    check("sigma 严格单调递减", bool((sigmas[1:] < sigmas[:-1]).all()))
    check("shift=1 时步长均匀 = 1/N",
          bool(torch.allclose(sigmas[:-1] - sigmas[1:], torch.full((50,), 0.02), atol=1e-6)))
    # shift != 1 的 warp：sigma' = shift*s / (1 + (shift-1)*s)，端点不变
    warped = flow_match_sigmas(4, 3.0)
    expected = torch.tensor([1.0, 3 * 0.75 / (1 + 2 * 0.75), 3 * 0.5 / (1 + 2 * 0.5),
                             3 * 0.25 / (1 + 2 * 0.25), 0.0])
    check("shift=3 的 sigma warp 与官方公式一致",
          bool(torch.allclose(warped, expected, atol=1e-6)))
    try:
        flow_match_sigmas(0)
        check("num_steps=0 被 ValueError 拦截", False)
    except ValueError:
        check("num_steps=0 被 ValueError 拦截", True)

    # --- 积分方向：官方 x <- x + (sigma - sigma_next) * v（旧实现为 -） ---
    latent_shape = (8, 64)
    cond = torch.ones(1, 4, 8)  # abs().mean() = 1 -> 速度场 = 全 1
    dit = MockFlowDiT()
    gen = torch.Generator(device="cpu").manual_seed(0)
    x_init = torch.randn((1, *latent_shape), generator=gen, dtype=torch.float32).squeeze(0)
    out = sample_latent_flow(
        dit, latent_shape, device, num_steps=10,
        cond=cond, guidance_scale=1.0,  # 关掉 CFG，单分支验证方向
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    check("输出形状 = latent_shape（已去 batch 维）", tuple(out.shape) == latent_shape)
    check("单分支积分 = x_T + (sigma_0 - sigma_N) * v（方向与官方一致）",
          bool(torch.allclose(out, x_init + 1.0, atol=1e-4)))
    check("guidance_scale<=1 时不做双分支前向", set(dit.batch_sizes) == {1})
    check("共走满 10 步", len(dit.batch_sizes) == 10)
    check("首步时间步 = sigma_0 × timestep_scale = 1000",
          abs(dit.timesteps[0] - 1000.0) < 1e-3)
    check("末步时间步 = sigma_{N-1} × 1000 = 100", abs(dit.timesteps[-1] - 100.0) < 1e-3)

    # --- CFG：batch=2 单次前向，v = v_uncond + gs * (v_cond - v_uncond) ---
    dit_cfg = MockFlowDiT()
    out_cfg = sample_latent_flow(
        dit_cfg, latent_shape, device, num_steps=10,
        cond=cond, guidance_scale=EVAL_GUIDANCE_SCALE,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    check("CFG 走 batch=2 单次前向（不是两次独立前向）",
          set(dit_cfg.batch_sizes) == {2} and len(dit_cfg.batch_sizes) == 10)
    # uncond 行为零嵌入 -> v_uncond = 0；v = gs * v_cond = 7
    check("CFG 组合 = uncond + 7.0 × (cond - uncond)",
          bool(torch.allclose(out_cfg, x_init + EVAL_GUIDANCE_SCALE, atol=1e-3)))
    check("cfg 强度 7.0 与推理侧一致", EVAL_GUIDANCE_SCALE == 7.0)

    # --- 条件全零（无 image_embeds 回退）：退化单分支，不付双倍开销 ---
    dit_zero = MockFlowDiT()
    sample_latent_flow(
        dit_zero, latent_shape, device, num_steps=3,
        cond=torch.zeros(1, 4, 8), guidance_scale=EVAL_GUIDANCE_SCALE,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    check("零条件退化为单分支采样", set(dit_zero.batch_sizes) == {1})

    # --- 无梯度 + 固定噪声可复现（best checkpoint 竞争需要可比分数）---
    check("采样结果不带梯度（@torch.no_grad）", not out.requires_grad)
    repeat = sample_latent_flow(
        MockFlowDiT(), latent_shape, device, num_steps=10,
        cond=cond, guidance_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    check("同种子 generator 采样逐位可复现", torch.equal(out, repeat))

    # --- shift 解析：优先读权重目录的 scheduler_config.json ---
    import json as _json

    cached = _EVAL_MODEL_CACHE.pop("flow_shift", None)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sched_dir = os.path.join(tmp_dir, "scheduler")
            os.makedirs(sched_dir)
            with open(os.path.join(sched_dir, "scheduler_config.json"), "w", encoding="utf-8") as h:
                _json.dump({"_class_name": "RectifiedFlowScheduler", "shift": 2.5}, h)
            shift = resolve_flow_shift({"model": {"diffusion": {"weights_path": tmp_dir}}})
            check("shift 取自权重目录 scheduler_config.json", abs(shift - 2.5) < 1e-9)
        _EVAL_MODEL_CACHE.pop("flow_shift", None)
        fallback = resolve_flow_shift({"model": {"diffusion": {"weights_path": ""}}})
        check("权重目录缺失时回落默认 shift", abs(fallback - EVAL_FLOW_SHIFT) < 1e-9)
    finally:
        _EVAL_MODEL_CACHE.pop("flow_shift", None)
        if cached is not None:
            _EVAL_MODEL_CACHE["flow_shift"] = cached


# ------------------------------------------------------------------ #
# 训练目标速度场符号（v1.5）
# ------------------------------------------------------------------ #
class MockConstDiT(nn.Module):
    """速度场恒为全 1（与输入无关）。

    这样 ``train_step`` 的损失可闭式推导为 ``mean((1 - velocity_target)^2)``，
    从而能从损失数值反解出它内部用的是哪个符号（不依赖读源码）。
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, timestep, encoder_hidden_states=None, return_dict=True):
        velocity = torch.ones_like(hidden_states)
        return (velocity,) if not return_dict else velocity


def test_velocity_target_sign() -> None:
    print("[h] 训练目标速度场符号（v = x₀ - ε）")
    torch.manual_seed(0)
    latents = torch.randn(2, 8, 64)

    # --- 1) 与基座 RectifiedFlowScheduler.step 的积分恒等式自洽 ---
    # 官方 step：x_next = x + (sigma - sigma_next) * v。把前向插值
    # x_sigma = (1 - sigma) * x₀ + sigma * ε 代入两端，v 必须 = x₀ - ε 才能精确成立。
    noise = torch.randn_like(latents)
    sigma, sigma_next = 0.8, 0.3
    x_sigma = (1.0 - sigma) * latents + sigma * noise
    x_sigma_next = (1.0 - sigma_next) * latents + sigma_next * noise
    v_new = latents - noise  # v1.5 约定
    v_old = noise - latents  # v1.0~v1.4 约定（反号）
    check(
        "v = x₀ - ε 精确满足官方 step 的积分恒等式",
        bool(torch.allclose(x_sigma + (sigma - sigma_next) * v_new, x_sigma_next, atol=1e-5)),
    )
    check(
        "v = ε - x₀（旧约定）不满足该恒等式",
        not bool(torch.allclose(x_sigma + (sigma - sigma_next) * v_old, x_sigma_next, atol=1e-3)),
    )

    # --- 2) train_step 实际使用的符号：用常量速度场从损失反解 ---
    dit = MockConstDiT()
    seed = 20260819
    torch.manual_seed(seed)
    loss = float(train_step(dit, latents, use_bf16=False).item())
    # 复现 train_step 内部的 RNG 消耗顺序（randn_like -> rand）以重建目标
    torch.manual_seed(seed)
    rep_noise = torch.randn_like(latents)
    _ = torch.rand(latents.shape[0])
    ones = torch.ones_like(latents)
    loss_new = float(((ones - (latents - rep_noise)) ** 2).mean().item())
    loss_old = float(((ones - (rep_noise - latents)) ** 2).mean().item())
    check("train_step 的 velocity_target = latents - noise", abs(loss - loss_new) < 1e-5)
    check(
        "train_step 的 velocity_target ≠ noise - latents（旧反号约定）",
        abs(loss - loss_old) > 1e-3,
    )


def main() -> None:
    torch.manual_seed(0)
    test_target_matching()
    test_param_groups()
    test_best_tracking()
    test_unconsumed_keys()
    test_mixed_cache_guard()
    test_lora_b_decay_validation()
    test_eval_sampler()
    test_velocity_target_sign()

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n===== {passed}/{total} 项测试通过 =====")
    if passed != total:
        for name, ok in RESULTS:
            if not ok:
                print(f"  失败: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()

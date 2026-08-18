"""v4 训练配方轻量单元测试（无 GPU，mock DiT 树）。

覆盖：
a) 新旧 target 匹配逻辑的挂载点数量（84 vs 42）与推理侧复现；
b) 参数组拆分：lora_B 带独立 weight_decay、lora_A 保持默认；缺省 0.0 时
   与旧行为（扁平参数列表）逐位一致；
c) BestCheckpointer 在模拟分数序列下的刷新逻辑（含 NaN 防护）；
d) load_lora_state_dict 未消费键校验；
e) LatentCacheDataset 混装缓存护栏；
f) lora_b_weight_decay 配置校验。

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
    BestCheckpointer,
    LatentCacheDataset,
    _match_target,
    build_lora_param_groups,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    parse_lora_b_weight_decay,
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


def main() -> None:
    torch.manual_seed(0)
    test_target_matching()
    test_param_groups()
    test_best_tracking()
    test_unconsumed_keys()
    test_mixed_cache_guard()
    test_lora_b_decay_validation()

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

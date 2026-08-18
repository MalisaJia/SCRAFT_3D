"""按 uid 白名单从源 latent 缓存子集出新缓存目录（类别专训用）。

源缓存目录布局（与 scripts/precompute_latents.py 的 flush_shard 一致）：
    latent_shard_*.pt   —— {'latents', 'captions', 'uids'[, 'image_embeds']}
    manifest.json       —— 元数据（可选，缺失时仅按 shard 文件扫描）

流程：逐 shard 加载 -> 按 uid 掩码（白名单交集）重堆叠 latents/captions/uids
（image_embeds 若存在则一并掩码）-> 跳过空 shard -> 按 --shard_size
重新分片写入目标目录的 latent_shard_*.pt -> 重写 manifest.json
（processed_uids 为子集排序列表，stats 注明 subset_of 源目录与白名单大小）。

uid 一律按 str 比较（防 int/str 不一致）。

用法：
    python scripts/subset_latent_cache.py \
        --src_cache /path/to/cache_full \
        --dst_cache /path/to/cache_chair \
        --whitelist /path/to/chair_uids.json \
        [--shard_size 256]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Set

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 uid 白名单从源 latent 缓存子集出新缓存目录"
    )
    parser.add_argument("--src_cache", type=str, required=True,
                        help="源缓存目录（含 latent_shard_*.pt）")
    parser.add_argument("--dst_cache", type=str, required=True,
                        help="输出缓存目录")
    parser.add_argument("--whitelist", type=str, required=True,
                        help="uid 白名单 JSON 文件（uid 字符串列表或 {uid: ...} 字典）")
    parser.add_argument("--shard_size", type=int, default=256,
                        help="输出 shard 的样本数上限（默认 256）")
    return parser.parse_args()


def load_uid_whitelist(path: str) -> Set[str]:
    """加载 uid 白名单 JSON（列表取元素、字典取键），uid 一律转为 str。"""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items = data.keys() if isinstance(data, dict) else data
    return {str(u) for u in items}


def list_src_shards(src_cache: str) -> List[str]:
    """按 shard 编号升序返回源目录的 latent_shard_*.pt 路径列表。"""
    names = [n for n in os.listdir(src_cache)
             if n.startswith("latent_shard_") and n.endswith(".pt")]

    def shard_key(name: str) -> int:
        try:
            return int(name[len("latent_shard_"): -len(".pt")])
        except ValueError:
            return 1 << 30  # 无法解析编号的排到最后

    return [os.path.join(src_cache, n) for n in sorted(names, key=shard_key)]


def flush_shard(buffer: Dict[str, Any], dst_cache: str, shard_index: int,
                manifest: Dict[str, Any]) -> str:
    """把缓冲样本写入 latent_shard_{i}.pt 并追加 manifest 的 shard 信息。"""
    shard_path = os.path.join(dst_cache, "latent_shard_%d.pt" % shard_index)
    shard = {
        "latents": torch.stack(buffer["latents"]),
        "captions": buffer["captions"],
        "uids": buffer["uids"],
    }
    embeds = buffer.get("image_embeds", [])
    if embeds and len(embeds) == len(buffer["uids"]):
        shard["image_embeds"] = torch.stack(embeds)
    torch.save(shard, shard_path)

    manifest["shards"].append({
        "path": os.path.basename(shard_path),
        "num_samples": len(buffer["uids"]),
        "uids": list(buffer["uids"]),
    })
    manifest["num_samples"] += len(buffer["uids"])

    buffer["latents"].clear()
    buffer["captions"].clear()
    buffer["uids"].clear()
    if "image_embeds" in buffer:
        buffer["image_embeds"].clear()
    return shard_path


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.src_cache):
        raise FileNotFoundError("源缓存目录不存在: %s" % args.src_cache)

    # ---- 启动防护：禁止覆盖源目录 / 已有内容的目标目录 ----
    src_abs = os.path.abspath(args.src_cache)
    dst_abs = os.path.abspath(args.dst_cache)
    if dst_abs == src_abs:
        raise ValueError("dst_cache 与 src_cache 相同，拒绝原地覆盖: %s" % src_abs)
    if os.path.isfile(os.path.join(dst_abs, "manifest.json")):
        raise FileExistsError(
            "dst_cache 已含 manifest.json，拒绝覆盖（请换新目录或先清理）: %s" % dst_abs)
    if os.path.isdir(dst_abs):
        stale = sorted(n for n in os.listdir(dst_abs)
                       if n.startswith("latent_shard_") and n.endswith(".pt"))
        if stale:
            raise FileExistsError(
                "dst_cache 已含 %d 个 latent_shard_*.pt（如 %s），拒绝覆盖: %s"
                % (len(stale), stale[0], dst_abs))
    os.makedirs(args.dst_cache, exist_ok=True)

    whitelist = load_uid_whitelist(args.whitelist)
    shard_paths = list_src_shards(args.src_cache)
    if not shard_paths:
        raise FileNotFoundError("源目录中没有 latent_shard_*.pt: %s" % args.src_cache)

    # ---- 源 manifest（可选，用于继承 latent_shape 等元数据）----
    src_manifest: Dict[str, Any] = {}
    src_manifest_path = os.path.join(args.src_cache, "manifest.json")
    if os.path.isfile(src_manifest_path):
        with open(src_manifest_path, "r", encoding="utf-8") as fh:
            src_manifest = json.load(fh)

    buffer: Dict[str, Any] = {"latents": [], "captions": [], "uids": [],
                              "image_embeds": []}
    manifest: Dict[str, Any] = {
        "num_samples": 0,
        "latent_shape": src_manifest.get("latent_shape"),
        "image_embed_shape": src_manifest.get("image_embed_shape"),
        "latent_dtype": src_manifest.get("latent_dtype", "float16"),
        "num_surface_points": src_manifest.get("num_surface_points"),
        "shards": [],
        "processed_uids": [],
        "stats": {},
    }

    seen_uids: Set[str] = set()
    src_samples = 0
    kept = 0
    num_empty_shards = 0
    shard_index = 0
    drop_embeds = False          # image_embeds 混合后全程放弃
    embed_state: bool | None = None      # 首个保留 shard 的 embeds 有无
    embed_state_shard = ""

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        uids = [str(u) for u in shard.get("uids", [])]
        src_samples += len(uids)
        has_embeds = "image_embeds" in shard

        # uid 掩码：在白名单内且未出现过（选中即记 seen，shard 内重复 uid 也只留一份）
        indices = []
        for i, uid in enumerate(uids):
            if uid in whitelist and uid not in seen_uids:
                seen_uids.add(uid)
                indices.append(i)
        if not indices:
            num_empty_shards += 1
            continue

        # ---- image_embeds 混合探测：有/无 embeds 的源 shard 混合则全程放弃 ----
        if not drop_embeds:
            if embed_state is None:
                embed_state = has_embeds
                embed_state_shard = os.path.basename(shard_path)
            elif has_embeds != embed_state:
                drop_embeds = True
                buffer["image_embeds"].clear()
                manifest["image_embed_shape"] = None
                print("[WARNING] 检测到 image_embeds 混合的源 shard：%s(%s) vs %s(%s)，"
                      "全程放弃 image_embeds（输出 shard 将不含该字段）"
                      % (embed_state_shard, "有" if embed_state else "无",
                         os.path.basename(shard_path), "有" if has_embeds else "无"),
                      flush=True)

        idx = torch.tensor(indices, dtype=torch.long)
        buffer["latents"].extend(list(shard["latents"][idx]))
        buffer["captions"].extend(shard["captions"][i] for i in indices)
        buffer["uids"].extend(uids[i] for i in indices)
        if not drop_embeds and has_embeds:
            buffer["image_embeds"].extend(list(shard["image_embeds"][idx]))
        kept += len(indices)

        # ---- 分片落盘 ----
        while len(buffer["uids"]) >= args.shard_size:
            head = {k: v[: args.shard_size] for k, v in buffer.items()}
            rest = {k: v[args.shard_size:] for k, v in buffer.items()}
            num_in_head = len(head["uids"])  # flush_shard 会清空 head 内列表，先记录
            path = flush_shard(head, args.dst_cache, shard_index, manifest)
            log("[落盘] %s（%d 样本）" % (path, num_in_head))
            shard_index += 1
            buffer = rest

    # ---- 空交集防护：与 precompute 侧"无交集即报错"行为一致 ----
    if kept == 0:
        raise RuntimeError(
            "白名单与源缓存无交集（%d 个白名单 uid 均未命中），未输出任何样本: "
            "src_cache=%s whitelist=%s"
            % (len(whitelist), os.path.abspath(args.src_cache),
               os.path.abspath(args.whitelist)))

    # ---- 尾片 ----
    if buffer["uids"]:
        num_in_tail = len(buffer["uids"])
        path = flush_shard(buffer, args.dst_cache, shard_index, manifest)
        log("[落盘] %s（尾片，%d 样本）" % (path, num_in_tail))
        shard_index += 1

    # ---- manifest ----
    manifest["processed_uids"] = sorted(seen_uids)
    manifest["stats"] = {
        "subset_of": os.path.abspath(args.src_cache),
        "whitelist": os.path.abspath(args.whitelist),
        "whitelist_size": len(whitelist),
        "source_samples": src_samples,
        "source_shards": len(shard_paths),
        "empty_shards_skipped": num_empty_shards,
        "intersection": kept,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path = os.path.join(args.dst_cache, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    log("[完成] 源样本数=%d（%d 个 shard，空 shard 跳过 %d）白名单数=%d "
        "交集数=%d 输出 shard 数=%d -> %s"
        % (src_samples, len(shard_paths), num_empty_shards, len(whitelist),
           kept, shard_index, args.dst_cache))


if __name__ == "__main__":
    main()

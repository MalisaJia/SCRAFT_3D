"""Objaverse 流式扩容预计算：边下载 glb 边编码 VAE latent + DINOv2 图像条件。

背景：GAN 时代的 200 个样本来自 ``/root/autodl-tmp/objaverse_data``，其下载方式为
按 ``hf-objaverse-v1/object-paths.json.gz``（uid -> glbs/<shard>/<uid>.glb）逐个从
HuggingFace ``allenai/objaverse`` resolve 直链下载 glb（无 tar、无断点残留）。
本脚本复用同款方式：逐个 uid 下载 glb 到暂存目录 -> trimesh 过滤 -> 归一化 ->
VAE encode latent [2048,64] fp16 -> 单视图渲染 512 -> DINOv2 image_embeds
[1370,1024] fp16 -> 追加 shard（与 train_diffusion.py 的 LatentCacheDataset 兼容），
处理完即删 glb；manifest 记录已尝试 uid，支持断点续跑。

用法（服务器上）：
    cd /root/autodl-tmp/3D-gans
    nohup /root/miniconda3/bin/python scripts/precompute_objaverse_stream.py \
        > logs/precompute_objaverse_stream.log 2>&1 &

红线遵守：不启动训练；每轮下载前检查磁盘余量（< 5GB 即停止并落盘）。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import shutil
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 复用 precompute_latents.py 的归一化 / 采样 / FPS 补丁 / shard 落盘逻辑
from scripts.precompute_latents import (  # noqa: E402
    LATENT_DTYPE,
    canonicalize_mesh,
    encode_mesh_latent,
    flush_shard,
    load_dinov2_encoder,
    load_triposg_vae,
    scan_existing_shards,
)
from rendering.multi_view_render import MultiViewRenderer, look_at_matrix  # noqa: E402

HF_BASES = (
    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/",
    "https://hf-mirror.com/datasets/allenai/objaverse/resolve/main/",
)
DEFAULT_OBJECT_PATHS = (
    "/root/autodl-tmp/objaverse_data/hf-objaverse-v1/object-paths.json.gz"
)
DEFAULT_OLD_ANNOTATIONS = "/root/autodl-tmp/objaverse_data/annotations.json"
DEFAULT_CACHE_DIR = "/root/autodl-tmp/3D-gans/cache/triposg_latents_objaverse"
DEFAULT_TMP_DIR = "/root/autodl-tmp/objaverse_stream_tmp"
DEFAULT_WEIGHTS = "/root/autodl-tmp/3D-gans/weights/triposg"
DEFAULT_CAPTION = "a 3D model of an object"
DINO_INPUT_SIZE = 518  # TripoSG pipeline 约定的 DINOv2 输入分辨率（37×37 patches）

MIN_FACES = 500
MAX_FACES = 300000
MIN_BBOX_SIDE = 1e-3
DISK_GUARD_BYTES = 5 * 1024 ** 3  # 余量低于 5GB 停止


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_uid_whitelist(path: str) -> set:
    """加载 uid 白名单 JSON（uid 字符串列表，或 {uid: ...} 字典取其键）。

    uid 一律转为 str 比较（防 int/str 不一致）。返回 str 集合。
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items = data.keys() if isinstance(data, dict) else data
    return {str(u) for u in items}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000,
                        help="good 样本目标数（达到即停）")
    parser.add_argument("--max_attempts", type=int, default=8000,
                        help="本次运行最多尝试的 uid 数（防低通过率无限跑）")
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--num_surface_points", type=int, default=8192)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tmp_dir", type=str, default=DEFAULT_TMP_DIR)
    parser.add_argument("--object_paths", type=str, default=DEFAULT_OBJECT_PATHS)
    parser.add_argument("--uid_whitelist", type=str, default=None,
                        help="uid 白名单 JSON 文件（uid 字符串列表或 {uid: ...} 字典）；"
                             "提供后仅处理白名单内的 uid（类别专训用）")
    parser.add_argument("--old_annotations", type=str,
                        default=DEFAULT_OLD_ANNOTATIONS)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--render_size", type=int, default=512)
    parser.add_argument("--prefetch", type=int, default=8,
                        help="下载预取并发数（暂存目录 glb 上限受此约束，远小于 200）")
    return parser.parse_args()


# ====================================================================== #
# 下载（GAN 时代同款：HF resolve 直链，逐文件）
# ====================================================================== #
def download_glb(rel_path: str, dest: str, retries: int = 3) -> bool:
    """下载单个 glb（带断点续传语义：.part 中转 + 多镜像重试）。"""
    part = dest + ".part"
    for attempt in range(retries):
        base = HF_BASES[attempt % len(HF_BASES)]
        url = base + rel_path
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            if os.path.isfile(part):
                req.add_header("Range", "bytes=%d-" % os.path.getsize(part))
            with urllib.request.urlopen(req, timeout=180) as resp:
                mode = "ab" if os.path.isfile(part) else "wb"
                with open(part, mode) as fh:
                    shutil.copyfileobj(resp, fh)
            os.replace(part, dest)
            return True
        except Exception as exc:  # noqa: BLE001
            log("[下载] 重试 %d %s: %s" % (attempt + 1, rel_path, exc))
            time.sleep(1.5 * (attempt + 1))
    if os.path.isfile(part):
        os.remove(part)
    return False


# ====================================================================== #
# glb -> (vertices, faces)，trimesh 加载 + scene 合并
# ====================================================================== #
def load_glb_mesh(glb_path: str) -> Tuple[np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(glb_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("scene 中无 mesh 几何（可能为点云/空场景）")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError("不支持的几何类型: %s" % type(loaded).__name__)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces


def mesh_passes_filter(vertices: np.ndarray, faces: np.ndarray) -> Optional[str]:
    """返回 None 表示通过，否则返回拒绝原因。"""
    if vertices.size == 0 or faces.size == 0:
        return "empty"
    if not np.isfinite(vertices).all():
        return "non-finite vertices"
    if len(faces) < MIN_FACES or len(faces) > MAX_FACES:
        return "face count %d out of [%d, %d]" % (len(faces), MIN_FACES, MAX_FACES)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    if extents.min() <= MIN_BBOX_SIDE:
        return "degenerate bbox %s" % np.array2string(extents, precision=5)
    return None


# ====================================================================== #
# 单视图渲染（确定性视角：azimuth 30° / elevation 20°）
# ====================================================================== #
@torch.no_grad()
def render_reference_image(
    renderer: MultiViewRenderer, vertices_np: np.ndarray, faces_np: np.ndarray,
    device: torch.device,
) -> Image.Image:
    vertices = torch.from_numpy(vertices_np).unsqueeze(0).to(device)  # [1,V,3]
    faces = torch.from_numpy(faces_np.astype(np.int64)).to(device)  # [F,3]

    az, el, dist = np.deg2rad(30.0), np.deg2rad(20.0), renderer.camera_distance
    eye = torch.tensor(
        [[dist * np.cos(el) * np.sin(az), dist * np.sin(el),
          dist * np.cos(el) * np.cos(az)]],
        dtype=torch.float32, device=device,
    )
    target = torch.zeros_like(eye)
    up = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    pose = look_at_matrix(eye, target, up)  # [1,4,4]

    out = renderer.render(vertices, faces, camera_poses=pose.unsqueeze(0))
    img_t = out["images"][0, 0].clamp(0.0, 1.0)  # [3,H,W]
    arr = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
def encode_image_embeds_pil(
    dinov2: Any, image: Image.Image, device: torch.device
) -> torch.Tensor:
    """DINOv2 + BitImageProcessor：last_hidden_state [1,1370,1024] -> fp16。

    权重目录里的 feature_extractor 预设为 224 输入（257 tokens），而
    TripoSG pipeline 约定 518×518（37×37=1369 patches + CLS = 1370 tokens，
    与训练侧 DIT_COND_NUM_TOKENS 一致），这里显式覆盖 resize/crop 尺寸。
    """
    processor, encoder = dinov2
    inputs = processor(
        images=image,
        return_tensors="pt",
        size={"height": DINO_INPUT_SIZE, "width": DINO_INPUT_SIZE},
        crop_size={"height": DINO_INPUT_SIZE, "width": DINO_INPUT_SIZE},
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with ctx:
        outputs = encoder(**inputs)
    embeds = outputs.last_hidden_state.squeeze(0).float()  # [S,1024]
    return embeds.to(LATENT_DTYPE).cpu()


# ====================================================================== #
# manifest（含已尝试 uid，断点续跑）
# ====================================================================== #
def load_or_init_manifest(cache_dir: str, num_surface_points: int) -> Dict[str, Any]:
    path = os.path.join(cache_dir, "manifest.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    else:
        manifest = {}
    manifest.setdefault("num_samples", 0)
    manifest.setdefault("latent_shape", None)
    manifest.setdefault("image_embed_shape", None)
    manifest.setdefault("latent_dtype", str(LATENT_DTYPE).split(".")[-1])
    manifest.setdefault("num_surface_points", num_surface_points)
    manifest.setdefault("shards", [])
    manifest.setdefault("processed_uids", [])
    manifest.setdefault("stats", {})
    return manifest


def save_manifest(manifest: Dict[str, Any], cache_dir: str) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False)


def free_bytes(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


# ====================================================================== #
# 主流程
# ====================================================================== #
def main() -> None:
    args = parse_args()

    # ---- 白名单模式防护：禁止把类别专训样本写进默认共享缓存目录 ----
    if args.uid_whitelist and (
        os.path.abspath(args.cache_dir) == os.path.abspath(DEFAULT_CACHE_DIR)
    ):
        raise RuntimeError(
            "白名单模式禁止写入默认共享缓存目录 %s，"
            "请显式指定独立的 --cache_dir（如类别专训目录）" % DEFAULT_CACHE_DIR)

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # ---- uid -> rel_path（GAN 时代同款 object-paths.json.gz）----
    with gzip.open(args.object_paths, "rt", encoding="utf-8") as fh:
        object_paths: Dict[str, str] = json.load(fh)
    log("[数据] object-paths 共 %d 个 uid" % len(object_paths))

    # ---- uid 白名单过滤（可选；作为 resume/processed 之外的额外过滤层）----
    if args.uid_whitelist:
        whitelist = load_uid_whitelist(args.uid_whitelist)
        num_before = len(object_paths)
        object_paths = {u: p for u, p in object_paths.items() if str(u) in whitelist}
        log("[白名单] %s 共 %d 个 uid，与索引交集 %d/%d（仅处理白名单内 uid）"
            % (args.uid_whitelist, len(whitelist), len(object_paths), num_before))
        if not object_paths:
            raise RuntimeError("白名单与 object-paths 索引无交集，无候选 uid，终止")

    # ---- 旧 annotations 的 caption ----
    old_captions: Dict[str, str] = {}
    if os.path.isfile(args.old_annotations):
        with open(args.old_annotations, "r", encoding="utf-8") as fh:
            for uid, info in json.load(fh).items():
                old_captions[uid] = info.get("caption", DEFAULT_CAPTION)

    # ---- 断点续跑：已处理（shard 扫描）+ 已尝试（manifest）----
    done_uids, next_shard_index = scan_existing_shards(args.cache_dir)
    manifest = load_or_init_manifest(args.cache_dir, args.num_surface_points)
    tried_uids = set(manifest.get("processed_uids", [])) | done_uids
    pending = [u for u in object_paths if u not in tried_uids]
    log("[续跑] 已完成 %d，已尝试 %d，本次候选 %d（shard 编号从 %d 起）"
        % (len(done_uids), len(tried_uids), len(pending), next_shard_index))

    # ---- 模型加载 ----
    log("[模型] 加载 TripoSG VAE ...")
    vae = load_triposg_vae(args.weights, device)
    dinov2 = load_dinov2_encoder(args.weights, device)
    if dinov2 is None:
        raise RuntimeError("image_encoder_dinov2 缺失，无法生成图像条件，终止")
    renderer = MultiViewRenderer(
        image_size=args.render_size, num_views=1, background=1.0, device=str(device)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log("[模型] VAE + DINOv2 + 渲染器就绪")

    buffer: Dict[str, Any] = {"latents": [], "captions": [], "uids": [],
                              "image_embeds": []}
    shard_index = next_shard_index
    good = failed_download = failed_filter = failed_encode = 0
    attempted = 0
    stop_reason = "target reached"

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.prefetch)

    def submit_batch(uids: List[str]) -> Dict[str, Any]:
        """提交一批下载任务，返回 uid -> future 映射。"""
        jobs = {}
        for uid in uids:
            rel = object_paths[uid]
            dest = os.path.join(args.tmp_dir, uid + ".glb")
            jobs[uid] = executor.submit(download_glb, rel, dest)
        return jobs

    queue: List[str] = []
    jobs: Dict[str, Any] = {}

    while good < args.target and attempted < args.max_attempts:
        # ---- 磁盘守卫 ----
        avail = free_bytes(args.cache_dir)
        if avail < DISK_GUARD_BYTES:
            stop_reason = "disk guard: free %.1fGB < 5GB" % (avail / 1024 ** 3)
            break

        # ---- 补充下载队列（暂存 glb 数始终 <= prefetch <= 200）----
        while len(queue) < args.prefetch and pending:
            batch = pending[: args.prefetch]
            pending = pending[args.prefetch:]
            jobs.update(submit_batch(batch))
            queue.extend(batch)
        if not queue:
            stop_reason = "uid exhausted"
            break

        uid = queue.pop(0)
        fut = jobs.pop(uid)
        attempted += 1
        tried_uids.add(uid)
        glb_path = os.path.join(args.tmp_dir, uid + ".glb")

        def cleanup() -> None:
            for suffix in ("", ".part"):
                p = glb_path + suffix
                if os.path.isfile(p):
                    os.remove(p)

        try:
            if not fut.result():
                failed_download += 1
                cleanup()
                continue

            # ---- trimesh 过滤 ----
            try:
                vertices, faces = load_glb_mesh(glb_path)
            except Exception as exc:  # noqa: BLE001
                failed_filter += 1
                log("[过滤] uid=%s 加载失败: %s" % (uid, exc))
                cleanup()
                continue
            reason = mesh_passes_filter(vertices, faces)
            if reason is not None:
                failed_filter += 1
                cleanup()
                continue

            # ---- 归一化 + VAE latent ----
            vertices_n, faces_n = canonicalize_mesh(vertices, faces)
            latent = encode_mesh_latent(
                vae, vertices_n, faces_n, device, args.num_surface_points
            )
            if manifest["latent_shape"] is None:
                manifest["latent_shape"] = list(latent.shape)
            elif list(latent.shape) != manifest["latent_shape"]:
                failed_encode += 1
                cleanup()
                continue

            # ---- 单视图渲染 + DINOv2 嵌入 ----
            image = render_reference_image(renderer, vertices_n, faces_n, device)
            embeds = encode_image_embeds_pil(dinov2, image, device)
            if manifest["image_embed_shape"] is None:
                manifest["image_embed_shape"] = list(embeds.shape)
            elif list(embeds.shape) != manifest["image_embed_shape"]:
                failed_encode += 1
                cleanup()
                continue

            buffer["latents"].append(latent)
            buffer["image_embeds"].append(embeds)
            buffer["captions"].append(old_captions.get(uid, DEFAULT_CAPTION))
            buffer["uids"].append(uid)
            good += 1
            cleanup()

            if good % 50 == 0:
                elapsed = time.time() - t_start
                log("[进度] good=%d/%d 尝试=%d 过滤失败=%d 下载失败=%d "
                    "编码失败=%d | %.2fs/样本 | 余量 %.1fGB"
                    % (good, args.target, attempted, failed_filter,
                       failed_download, failed_encode, elapsed / max(good, 1),
                       free_bytes(args.cache_dir) / 1024 ** 3))

            # ---- shard 落盘 ----
            if len(buffer["uids"]) >= args.shard_size:
                shard_path = flush_shard(buffer, args.cache_dir, shard_index,
                                         manifest)
                manifest["processed_uids"] = sorted(tried_uids)
                manifest["stats"] = {
                    "good": good, "attempted": attempted,
                    "failed_download": failed_download,
                    "failed_filter": failed_filter,
                    "failed_encode": failed_encode,
                }
                save_manifest(manifest, args.cache_dir)
                log("[落盘] %s（good 累计 %d）" % (shard_path, manifest["num_samples"]))
                shard_index += 1

        except Exception as exc:  # noqa: BLE001
            failed_encode += 1
            log("[编码] uid=%s 失败: %s" % (uid, exc))
            cleanup()
            continue
        finally:
            # 每 200 次尝试刷新一次 manifest 的已尝试集合（断点续跑）
            if attempted % 200 == 0:
                manifest["processed_uids"] = sorted(tried_uids)
                save_manifest(manifest, args.cache_dir)

    # ---- 尾片与收尾 ----
    executor.shutdown(wait=False, cancel_futures=True)
    if buffer["uids"]:
        shard_path = flush_shard(buffer, args.cache_dir, shard_index, manifest)
        log("[落盘] %s（尾片，good 累计 %d）" % (shard_path, manifest["num_samples"]))
    manifest["processed_uids"] = sorted(tried_uids)
    manifest["stats"] = {
        "good": good, "attempted": attempted,
        "failed_download": failed_download, "failed_filter": failed_filter,
        "failed_encode": failed_encode, "stop_reason": stop_reason,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "seconds_per_good": round((time.time() - t_start) / max(good, 1), 2),
    }
    if device.type == "cuda":
        manifest["stats"]["vram_peak_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024 ** 3, 2
        )
    save_manifest(manifest, args.cache_dir)

    pass_rate = 100.0 * good / max(attempted, 1)
    log("[完成] stop=%s good=%d 尝试=%d 通过率=%.1f%% | 下载失败=%d 过滤失败=%d "
        "编码失败=%d | 耗时 %.0fs（%.2fs/样本）| 缓存共 %d 样本 -> %s"
        % (stop_reason, good, attempted, pass_rate, failed_download,
           failed_filter, failed_encode, time.time() - t_start,
           (time.time() - t_start) / max(good, 1), manifest["num_samples"],
           args.cache_dir))


if __name__ == "__main__":
    main()

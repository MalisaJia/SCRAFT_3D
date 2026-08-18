# -*- coding: utf-8 -*-
"""生成 ShapeNet annotations JSON（与 datasets/objaverse.py 的解析格式一致）。

遍历 /root/autodl-tmp/data/shapenet/<synset>/<md5>/models/model_normalized.obj，
写出 {uid: {"path", "category", "caption"}} 形式的 JSON，供
scripts/precompute_latents.py --config 中 data.annotations_path 指向使用
（data_root 传 /root/autodl-tmp/data/shapenet 或直接用绝对 path）。

用法：
    python scripts/make_shapenet_annotations.py \
        [--data_root /root/autodl-tmp/data/shapenet] \
        [--out /root/autodl-tmp/data/shapenet_annotations.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.shapenet import SHAPENET_SYNSET_IDS  # noqa: E402

SYNSET_TO_NAME = {v: k for k, v in SHAPENET_SYNSET_IDS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", default="/root/autodl-tmp/data/shapenet"
    )
    parser.add_argument(
        "--out", default="/root/autodl-tmp/data/shapenet_annotations.json"
    )
    args = parser.parse_args()

    annotations: dict = {}
    per_category: dict = {}
    for synset, name in sorted(SYNSET_TO_NAME.items()):
        cat_dir = os.path.join(args.data_root, synset)
        if not os.path.isdir(cat_dir):
            continue
        for model_id in sorted(os.listdir(cat_dir)):
            obj = os.path.join(cat_dir, model_id, "models", "model_normalized.obj")
            if not os.path.isfile(obj):
                # 兼容非官方布局：任意 *.obj
                fallback = None
                for root, _dirs, files in os.walk(os.path.join(cat_dir, model_id)):
                    for fname in files:
                        if fname.lower().endswith(".obj"):
                            fallback = os.path.join(root, fname)
                            break
                    if fallback:
                        break
                if fallback is None:
                    continue
                obj = fallback
            caption = "a chair" if name == "chair" else f"a 3D model of {name}"
            uid = f"{synset}-{model_id}"
            annotations[uid] = {
                "path": obj,
                "category": name,
                "caption": caption,
            }
            per_category[name] = per_category.get(name, 0) + 1

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(annotations, fh, ensure_ascii=False, indent=2)

    print(f"written {len(annotations)} entries -> {args.out}")
    for name, count in sorted(per_category.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

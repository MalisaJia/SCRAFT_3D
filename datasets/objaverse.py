"""Objaverse 3D model dataset loader.

Loads meshes from an Objaverse-style dump together with their textual
descriptions. Supports filtering to the LVIS category subset and lazily loads
mesh geometry on demand (the corpus is far too large to hold in memory).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .shapenet import load_obj


class ObjaverseDataset(Dataset):
    """Objaverse mesh dataset with lazy loading.

    The dataset is described by a JSON annotation file mapping each object uid
    to its metadata. The expected schema (extra keys are ignored) is::

        {
            "<uid>": {
                "path": "relative/or/absolute/path/to/model.obj",
                "category": "chair",          # optional
                "caption": "a wooden chair",  # optional free-form text
                "lvis": "chair"               # optional LVIS category
            },
            ...
        }

    Args:
        data_root: Root directory that mesh ``path`` entries are resolved
            against when they are relative.
        annotation_file: Path to the JSON annotation file described above. If
            ``None`` it defaults to ``<data_root>/annotations.json``.
        lvis_categories: Optional list of LVIS category names to keep. When
            provided, only objects whose ``lvis``/``category`` field matches
            one of these names are retained.
        num_points: If given, vertices are resampled to exactly this many
            points to yield fixed-size batches.
        caption_template: Format string used to synthesise a caption from a
            category name when an explicit ``caption`` is missing.
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: Optional[str] = None,
        lvis_categories: Optional[List[str]] = None,
        num_points: Optional[int] = None,
        caption_template: str = "a {}",
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.annotation_file = annotation_file or os.path.join(
            data_root, "annotations.json"
        )
        self.lvis_categories = (
            {c.lower() for c in lvis_categories} if lvis_categories else None
        )
        self.num_points = num_points
        self.caption_template = caption_template

        # Only lightweight metadata is materialised here; meshes are read from
        # disk lazily inside ``__getitem__``.
        self.entries: List[Dict[str, str]] = self._load_annotations()

        categories = sorted({e["category"] for e in self.entries if e["category"]})
        self.category_to_label: Dict[str, int] = {
            name: idx for idx, name in enumerate(categories)
        }

    # ------------------------------------------------------------------ #
    # Annotation loading / filtering
    # ------------------------------------------------------------------ #
    def _load_annotations(self) -> List[Dict[str, str]]:
        """Read and filter the annotation file into a flat entry list."""
        if not os.path.isfile(self.annotation_file):
            return []

        with open(self.annotation_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        entries: List[Dict[str, str]] = []
        for uid, meta in raw.items():
            category = str(meta.get("lvis") or meta.get("category") or "")
            if self.lvis_categories is not None:
                if category.lower() not in self.lvis_categories:
                    continue

            caption = meta.get("caption")
            if not caption:
                caption = (
                    self.caption_template.format(category)
                    if category
                    else "a 3d object"
                )

            entries.append(
                {
                    "uid": uid,
                    "path": str(meta.get("path", "")),
                    "category": category,
                    "caption": str(caption),
                }
            )
        return entries

    def _resolve_path(self, path: str) -> str:
        """Turn a possibly relative mesh path into an absolute one."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_root, path)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(vertices: np.ndarray) -> np.ndarray:
        """Center vertices and scale them into the unit sphere."""
        if vertices.size == 0:
            return vertices
        centroid = vertices.mean(axis=0, keepdims=True)
        vertices = vertices - centroid
        radius = np.linalg.norm(vertices, axis=1).max()
        if radius > 0:
            vertices = vertices / radius
        return vertices.astype(np.float32)

    def _resample(self, vertices: np.ndarray) -> np.ndarray:
        """Resample the vertex set to ``self.num_points`` rows."""
        if self.num_points is None or vertices.size == 0:
            return vertices
        replace = len(vertices) < self.num_points
        idx = np.random.choice(len(vertices), self.num_points, replace=replace)
        return vertices[idx]

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self.entries[index]
        mesh_path = self._resolve_path(entry["path"])

        # Lazy load: geometry is only touched when the sample is requested.
        if os.path.isfile(mesh_path):
            vertices, faces = load_obj(mesh_path)
        else:
            vertices = np.zeros((0, 3), dtype=np.float32)
            faces = np.zeros((0, 3), dtype=np.int64)

        vertices = self._normalize(vertices)
        vertices = self._resample(vertices)

        label = self.category_to_label.get(entry["category"], -1)
        return {
            "vertices": torch.from_numpy(vertices).float(),
            "faces": torch.from_numpy(faces).long(),
            "label": torch.tensor(label, dtype=torch.long),
            "category": entry["category"],
            "text": entry["caption"],
            "uid": entry["uid"],
        }

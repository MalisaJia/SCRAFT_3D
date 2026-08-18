"""ShapeNet 3D mesh dataset loader.

Loads triangle meshes stored in the Wavefront ``.obj`` format, optionally
filtered by semantic category, and applies light geometric data augmentation
(random rotation / scaling) for GAN training.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Mapping between human readable ShapeNet category names and their
# ShapeNetCore synset ids. Only a small, commonly used subset is listed here;
# unknown names are treated as raw directory names.
SHAPENET_SYNSET_IDS: Dict[str, str] = {
    "airplane": "02691156",
    "bench": "02828884",
    "cabinet": "02933112",
    "car": "02958343",
    "chair": "03001627",
    "display": "03211117",
    "lamp": "03636649",
    "loudspeaker": "03691459",
    "rifle": "04090263",
    "sofa": "04256520",
    "table": "04379243",
    "telephone": "04401088",
    "watercraft": "04530566",
}


def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a Wavefront ``.obj`` file.

    Args:
        path: Path to the ``.obj`` file.

    Returns:
        A tuple ``(vertices, faces)`` where ``vertices`` has shape
        ``(V, 3)`` (float32) and ``faces`` has shape ``(F, 3)`` (int64,
        zero-based indices).
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Faces may be encoded as "v", "v/vt" or "v/vt/vn"; keep the
                # vertex index only and convert to a zero-based index.
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                # Triangulate polygons using a simple fan triangulation.
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
    )


class ShapeNetDataset(Dataset):
    """ShapeNet mesh dataset.

    Args:
        data_root: Root directory of the ShapeNet dump. Each category is
            expected to live in its own sub-directory (either the synset id or
            the plain category name), containing one folder per model with a
            ``model.obj`` (or ``*.obj``) mesh.
        split: One of ``"train"``, ``"val"`` or ``"test"``. Splits are derived
            deterministically from a per-category model ordering.
        categories: Optional list of category names to include. When ``None``
            all discovered categories are used.
        augment: Whether to apply random rotation / scaling augmentation.
        num_points: If given, vertices are resampled (with replacement when
            necessary) to exactly this many points, producing fixed-size
            batches. When ``None`` the raw vertex count is preserved.
        rotation_axis: Axis around which random rotations are applied
            (``"x"``, ``"y"`` or ``"z"``).
        scale_range: ``(min, max)`` uniform scale factor sampled during
            augmentation.
        split_ratios: ``(train, val, test)`` fractions, must sum to 1.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        augment: bool = True,
        num_points: Optional[int] = None,
        rotation_axis: str = "y",
        scale_range: Tuple[float, float] = (0.8, 1.2),
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        if rotation_axis not in ("x", "y", "z"):
            raise ValueError(f"Unknown rotation axis: {rotation_axis!r}")

        self.data_root = data_root
        self.split = split
        self.augment = augment and split == "train"
        self.num_points = num_points
        self.rotation_axis = rotation_axis
        self.scale_range = scale_range
        self.split_ratios = split_ratios

        self.categories = self._resolve_categories(categories)
        self.category_to_label: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.categories)
        }
        self.samples: List[Tuple[str, str]] = self._index_samples()

    # ------------------------------------------------------------------ #
    # Indexing helpers
    # ------------------------------------------------------------------ #
    def _resolve_categories(self, categories: Optional[List[str]]) -> List[str]:
        """Return the sorted list of category names to load."""
        if categories is not None:
            return list(categories)
        if not os.path.isdir(self.data_root):
            return []
        found = [
            name
            for name in sorted(os.listdir(self.data_root))
            if os.path.isdir(os.path.join(self.data_root, name))
        ]
        return found

    def _category_dir(self, category: str) -> Optional[str]:
        """Locate the directory for a category (by synset id or name)."""
        candidates = [category, SHAPENET_SYNSET_IDS.get(category, "")]
        for candidate in candidates:
            if not candidate:
                continue
            path = os.path.join(self.data_root, candidate)
            if os.path.isdir(path):
                return path
        return None

    def _index_samples(self) -> List[Tuple[str, str]]:
        """Build the ``(obj_path, category)`` list for the requested split."""
        samples: List[Tuple[str, str]] = []
        for category in self.categories:
            cat_dir = self._category_dir(category)
            if cat_dir is None:
                continue

            model_paths = self._find_meshes(cat_dir)
            model_paths.sort()

            train_end = int(len(model_paths) * self.split_ratios[0])
            val_end = train_end + int(len(model_paths) * self.split_ratios[1])
            if self.split == "train":
                selected = model_paths[:train_end]
            elif self.split == "val":
                selected = model_paths[train_end:val_end]
            else:
                selected = model_paths[val_end:]

            samples.extend((path, category) for path in selected)
        return samples

    @staticmethod
    def _find_meshes(cat_dir: str) -> List[str]:
        """Recursively collect all ``.obj`` files below ``cat_dir``."""
        meshes: List[str] = []
        for root, _dirs, files in os.walk(cat_dir):
            for name in files:
                if name.lower().endswith(".obj"):
                    meshes.append(os.path.join(root, name))
        return meshes

    # ------------------------------------------------------------------ #
    # Augmentation
    # ------------------------------------------------------------------ #
    def _rotation_matrix(self, angle: float) -> np.ndarray:
        """Build a rotation matrix around ``self.rotation_axis``."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        if self.rotation_axis == "x":
            return np.array(
                [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]],
                dtype=np.float32,
            )
        if self.rotation_axis == "y":
            return np.array(
                [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]],
                dtype=np.float32,
            )
        return np.array(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
            dtype=np.float32,
        )

    def _augment(self, vertices: np.ndarray) -> np.ndarray:
        """Apply random rotation and scaling to ``vertices``."""
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        vertices = vertices @ self._rotation_matrix(angle).T
        scale = np.random.uniform(*self.scale_range)
        return (vertices * scale).astype(np.float32)

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
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        obj_path, category = self.samples[index]
        vertices, faces = load_obj(obj_path)

        vertices = self._normalize(vertices)
        if self.augment:
            vertices = self._augment(vertices)
        vertices = self._resample(vertices)

        return {
            "vertices": torch.from_numpy(vertices).float(),
            "faces": torch.from_numpy(faces).long(),
            "label": torch.tensor(self.category_to_label[category], dtype=torch.long),
            "category": category,
        }

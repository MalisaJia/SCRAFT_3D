"""3D mesh datasets for training."""

from .shapenet import ShapeNetDataset
from .objaverse import ObjaverseDataset

__all__ = ["ShapeNetDataset", "ObjaverseDataset"]

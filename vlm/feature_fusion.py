"""特征融合模块 + 视角感知文本描述生成。

视角感知文本是本方案的核心创新之一：渲染出的每个视角都配一条与其方位匹配的
文本描述（"a dog from the back" 等），从而让 CLIP 语义监督具备 3D 视角一致性，
避免所有视角都被单一 prompt 拉向同一个"正面"外观。
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class ViewAwarePromptGenerator:
    """根据相机视角自动生成视角感知的文本描述。

    核心创新：不使用单一 prompt，而是根据渲染角度生成方位描述
    例如：azimuth=0°   -> "a {category} from the front"
         azimuth=90°  -> "a {category} from the right side"
         azimuth=180° -> "a {category} from the back"
         elevation=45° -> "a {category} from above"
    """

    def __init__(
        self,
        base_template: str = "a {category} from the {azimuth}",
        include_elevation: bool = True,
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        """
        Args:
            base_template: 基础模板，需含 {category} 与 {azimuth} 占位符。
            include_elevation: 是否在描述中附加仰角短语。
            prefix: 追加在描述最前的内容，如 "a 3D rendering of "。
            suffix: 追加在描述末尾的内容，如 ", photorealistic"。
        """
        # 方位到文本的映射模板，区间为 [start, end) 的方位角度数
        self.azimuth_templates: Dict[str, Tuple[float, float]] = {
            "front": (315, 45),  # -45° to 45°，跨越 0° 需要环绕判断
            "right side": (45, 135),
            "back": (135, 225),
            "left side": (225, 315),
        }
        self.elevation_templates: Dict[str, Tuple[float, float]] = {
            "above": (20, 90),
            "slightly above": (5, 20),
            "level": (-5, 5),
            "slightly below": (-20, -5),
            "below": (-90, -20),
        }

        self.base_template = base_template
        self.include_elevation = include_elevation
        self.prefix = prefix
        self.suffix = suffix

    # ------------------------------------------------------------------ #
    # 单视角描述
    # ------------------------------------------------------------------ #
    def azimuth_to_text(self, azimuth: float) -> str:
        """把方位角（度）映射为方位词。"""
        azimuth = float(azimuth) % 360.0
        for name, (start, end) in self.azimuth_templates.items():
            if start <= end:
                if start <= azimuth < end:
                    return name
            else:  # 跨越 0°/360° 的区间，如 front (315, 45)
                if azimuth >= start or azimuth < end:
                    return name
        return "front"

    def elevation_to_text(self, elevation: float) -> str:
        """把仰角（度）映射为高度词，"level" 表示水平视角。"""
        elevation = float(elevation)
        elevation = max(-90.0, min(90.0, elevation))
        for name, (low, high) in self.elevation_templates.items():
            if low <= elevation < high:
                return name
        return "above" if elevation > 0 else "below"

    def describe_view(
        self, category: str, azimuth: float, elevation: float
    ) -> str:
        """生成单个视角的文本描述。

        Args:
            category: 物体类别，如 "dog"。
            azimuth: 方位角（度）。
            elevation: 仰角（度）。

        Returns:
            如 "a dog from the right side, slightly above"。
        """
        azimuth_word = self.azimuth_to_text(azimuth)
        prompt = self.base_template.format(category=category, azimuth=azimuth_word)

        if self.include_elevation:
            elevation_word = self.elevation_to_text(elevation)
            if elevation_word != "level":  # 水平视角不额外描述，保持 prompt 简洁
                prompt = f"{prompt}, {elevation_word}"

        return f"{self.prefix}{prompt}{self.suffix}"

    # ------------------------------------------------------------------ #
    # 批量描述
    # ------------------------------------------------------------------ #
    def generate_prompts(
        self,
        category: Union[str, List[str]],
        azimuths: Tensor,
        elevations: Tensor,
    ) -> List[str]:
        """根据类别和视角生成描述。

        Args:
            category: 物体类别 "dog"、"chair" 等；也可传长度为 B 的类别列表，
                对应 batch 内每个样本各自的类别。
            azimuths: [B, N_views] 方位角（度数）
            elevations: [B, N_views] 仰角（度数）

        Returns:
            List of prompts，长度 B*N_views，按 (batch, view) 行优先展开
            如 ["a dog from the front", "a dog from the right side, slightly above", ...]
        """
        if azimuths.shape != elevations.shape:
            raise ValueError(
                f"azimuths 与 elevations 形状需一致，实际为 "
                f"{tuple(azimuths.shape)} 与 {tuple(elevations.shape)}"
            )
        if azimuths.dim() == 1:  # 容忍 [N_views] 输入
            azimuths = azimuths.unsqueeze(0)
            elevations = elevations.unsqueeze(0)

        batch_size, num_views = azimuths.shape
        if isinstance(category, str):
            categories = [category] * batch_size
        else:
            if len(category) != batch_size:
                raise ValueError(
                    f"类别列表长度 {len(category)} 与 batch 大小 {batch_size} 不匹配"
                )
            categories = list(category)

        azimuth_list = azimuths.detach().float().cpu().tolist()
        elevation_list = elevations.detach().float().cpu().tolist()

        prompts: List[str] = []
        for b in range(batch_size):
            for v in range(num_views):
                prompts.append(
                    self.describe_view(
                        categories[b], azimuth_list[b][v], elevation_list[b][v]
                    )
                )
        return prompts

    def generate_prompts_from_radians(
        self,
        category: Union[str, List[str]],
        azimuths: Tensor,
        elevations: Tensor,
    ) -> List[str]:
        """与 generate_prompts 相同，但输入角度为弧度。

        渲染器返回的是弧度，训练脚本可直接调用本方法衔接。
        """
        return self.generate_prompts(
            category, torch.rad2deg(azimuths), torch.rad2deg(elevations)
        )

    def all_view_prompts(self, category: str) -> List[str]:
        """列出该类别在所有方位词下的描述，可用于调试与检索评估。"""
        return [
            self.base_template.format(category=category, azimuth=word)
            for word in self.azimuth_templates
        ]


class FeatureFusion(nn.Module):
    """特征融合模块：将几何特征和语义特征结合。

    v1 策略：分支独立 + 损失层加权（不在特征层面融合）
    提供接口供 v2 升级为注意力融合
    """

    def __init__(
        self,
        geometric_dim: int = 512,
        semantic_dim: int = 512,
        fusion_strategy: str = "separate",
        fusion_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            geometric_dim: 几何特征维度。
            semantic_dim: 语义（CLIP）特征维度。
            fusion_strategy: "separate"（v1，无参数直通）、"concat"（MLP 融合）、
                "attention"（v2，几何 query 与语义 key/value 的交叉注意力）。
            fusion_dim: 融合特征维度，默认取 geometric_dim。
            num_heads: attention 策略的多头数量。
            dropout: attention / MLP 的 dropout 比例。
        """
        super().__init__()
        if fusion_strategy not in ("separate", "concat", "attention"):
            raise ValueError(f"未知的 fusion_strategy: {fusion_strategy}")

        self.geometric_dim = geometric_dim
        self.semantic_dim = semantic_dim
        self.fusion_strategy = fusion_strategy
        self.fusion_dim = fusion_dim or geometric_dim

        if fusion_strategy == "concat":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(geometric_dim + semantic_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.fusion_dim, self.fusion_dim),
            )
        elif fusion_strategy == "attention":
            self.geo_proj = nn.Linear(geometric_dim, self.fusion_dim)
            self.sem_proj = nn.Linear(semantic_dim, self.fusion_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(self.fusion_dim)

    @property
    def produces_fused(self) -> bool:
        """当前策略是否输出 'fused' 特征。"""
        return self.fusion_strategy != "separate"

    def forward(
        self, geometric_features: Tensor, semantic_features: Tensor
    ) -> Dict[str, Tensor]:
        """
        Args:
            geometric_features: [B, geometric_dim] 或 [B, T, geometric_dim]
            semantic_features: [B, semantic_dim] 或 [B, T, semantic_dim]

        Returns:
            dict with 'geometric', 'semantic', and optionally 'fused' features
        """
        if geometric_features.shape[-1] != self.geometric_dim:
            raise ValueError(
                f"几何特征维度应为 {self.geometric_dim}，实际为 {geometric_features.shape[-1]}"
            )
        if semantic_features.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"语义特征维度应为 {self.semantic_dim}，实际为 {semantic_features.shape[-1]}"
            )

        outputs: Dict[str, Tensor] = {
            "geometric": geometric_features,
            "semantic": semantic_features,
        }

        # v1：两个分支保持独立，融合完全交给损失层的加权求和
        if self.fusion_strategy == "separate":
            return outputs

        if self.fusion_strategy == "concat":
            outputs["fused"] = self.fusion_mlp(
                torch.cat([geometric_features, semantic_features], dim=-1)
            )
            return outputs

        # attention：几何特征作 query，语义特征作 key/value，残差回加几何分支
        squeeze = geometric_features.dim() == 2
        geo = self.geo_proj(geometric_features)
        sem = self.sem_proj(semantic_features)
        if squeeze:
            geo = geo.unsqueeze(1)
            sem = sem.unsqueeze(1)

        attended, _ = self.attention(geo, sem, sem, need_weights=False)
        fused = self.norm(geo + attended)
        outputs["fused"] = fused.squeeze(1) if squeeze else fused
        return outputs

    def extra_repr(self) -> str:
        return (
            f"geometric_dim={self.geometric_dim}, semantic_dim={self.semantic_dim}, "
            f"fusion_strategy={self.fusion_strategy}, fusion_dim={self.fusion_dim}"
        )

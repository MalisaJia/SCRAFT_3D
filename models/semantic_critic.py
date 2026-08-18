"""语义合理性判别器 (Semantic Critic)。

在 CLIP 特征空间中**学习**区分正常结构与异常结构（多头、多腿、不自然
突起、对称性破坏等）。这里的判别能力来自有监督训练——正样本是真实
mesh，负样本是 :class:`~utils.anomaly_constructor.AnomalyConstructor`
程序化构造的异常 mesh——而不是 CLIP 的零样本推理：CLIP 只提供物体级
的语义特征表示，本身无法判断「八条腿的狗不合理」这类结构问题。因此
Critic 是对 CLIP 能力的增强与补充，而非依赖 CLIP 的常识。

相应地，Critic 的判别范围受训练时异常类型覆盖度的限制，不构成开放式
的常识推理。

与 :class:`~models.discriminator.UnifiedDiscriminator` 的语义头不同：
语义头只负责把渲染图对齐到 CLIP 文本空间（「像不像椅子」），而 Critic
接收 [CLIP 语义 ‖ 几何统计] 的联合输入，判别的是「这个结构本身是否合理」
（三条腿的椅子在 CLIP 空间里依然很像椅子，但几何统计会暴露异常）。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import spectral_norm

__all__ = ["SemanticCritic"]


class SemanticCritic(nn.Module):
    """语义合理性判别器。

    输入: CLIP image embedding + 几何特征
    输出: plausibility score ∈ [0, 1]（1 = 合理，0 = 异常）

    打分含义是「与训练时见过的正常结构分布相符的程度」，由异常负样本
    监督学习得到，不是 CLIP 的开放式常识判断。

    架构::

        [clip_feat ‖ geo_feat] -> Linear(in, hidden) -> LeakyReLU ->
        Linear(hidden, hidden//2) -> LeakyReLU -> Linear(hidden//2, 1) -> Sigmoid

    所有线性层使用 Spectral Normalization 稳定训练：Critic 的输入是外部
    网络（CLIP / 几何统计）给出的特征，尺度不受自身控制，谱归一化把
    Lipschitz 常数约束住，配合梯度惩罚避免打分尺度失控。

    Args:
        clip_dim: CLIP 图像 embedding 维度。
        geo_dim: 几何特征维度。
        hidden_dim: 第一层隐藏宽度（第二层为 ``hidden_dim // 2``）。
        dropout: 隐藏层 dropout 比例，<=0 时关闭。
    """

    def __init__(
        self,
        clip_dim: int = 512,
        geo_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if clip_dim < 1 or geo_dim < 1:
            raise ValueError(f"clip_dim / geo_dim 必须为正，得到 {clip_dim} / {geo_dim}")
        if hidden_dim < 2:
            raise ValueError(f"hidden_dim 至少为 2，得到 {hidden_dim}")

        self.clip_dim = clip_dim
        self.geo_dim = geo_dim
        self.hidden_dim = hidden_dim

        in_dim = clip_dim + geo_dim
        mid_dim = hidden_dim // 2

        layers: List[nn.Module] = [
            spectral_norm(nn.Linear(in_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [
            spectral_norm(nn.Linear(hidden_dim, mid_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(spectral_norm(nn.Linear(mid_dim, 1)))

        # backbone 输出 logits，sigmoid 单独持有，便于按需取未激活的打分
        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def _prepare(self, clip_features: Tensor, geo_features: Tensor) -> Tensor:
        """校验并拼接两路特征，返回 [B, clip_dim + geo_dim]。"""
        if clip_features.dim() != 2 or clip_features.shape[1] != self.clip_dim:
            raise ValueError(
                f"clip_features 形状应为 [B, {self.clip_dim}]，"
                f"实际为 {tuple(clip_features.shape)}"
            )
        if geo_features.dim() != 2 or geo_features.shape[1] != self.geo_dim:
            raise ValueError(
                f"geo_features 形状应为 [B, {self.geo_dim}]，"
                f"实际为 {tuple(geo_features.shape)}"
            )
        if clip_features.shape[0] != geo_features.shape[0]:
            raise ValueError(
                "clip_features 与 geo_features 的 batch 不一致："
                f"{clip_features.shape[0]} vs {geo_features.shape[0]}"
            )
        geo_features = geo_features.to(clip_features.dtype)
        return torch.cat([clip_features, geo_features], dim=-1)

    def forward(
        self,
        clip_features: Tensor,
        geo_features: Tensor,
        return_logits: bool = False,
    ) -> Tensor:
        """
        Args:
            clip_features: (B, clip_dim) CLIP 图像编码
            geo_features: (B, geo_dim) 几何特征（来自Discriminator的geometric head或独立提取）
            return_logits: 为 True 时返回未过 sigmoid 的 logits（BCEWithLogits /
                梯度惩罚使用该路径，数值更稳定）

        Returns:
            score: (B, 1) plausibility score（``return_logits=True`` 时为 logits）
        """
        logits = self.net(self._prepare(clip_features, geo_features))
        return logits if return_logits else torch.sigmoid(logits)

    def score(self, clip_features: Tensor, geo_features: Tensor) -> Tensor:
        """便捷接口：返回 [B, 1] 的 plausibility 概率。"""
        return self.forward(clip_features, geo_features, return_logits=False)

    # ------------------------------------------------------------------ #
    # 训练稳定性
    # ------------------------------------------------------------------ #
    def compute_gradient_penalty(
        self,
        real_clip: Tensor,
        real_geo: Tensor,
        fake_clip: Tensor,
        fake_geo: Tensor,
    ) -> Tensor:
        """计算梯度惩罚用于训练稳定性。

        WGAN-GP 形式：在正常样本与异常样本的连线上随机取插值点，惩罚
        Critic 对输入（CLIP 与几何两路一并计入）梯度模长偏离 1 的程度。

        两路输入使用同一组插值系数，保证插值点仍对应一个「语义-几何」配对，
        而不是把某个样本的 CLIP 特征与另一个样本的几何特征混在一起。

        Args:
            real_clip: (B, clip_dim) 正常样本的 CLIP 特征
            real_geo: (B, geo_dim) 正常样本的几何特征
            fake_clip: (B', clip_dim) 异常样本的 CLIP 特征
            fake_geo: (B', geo_dim) 异常样本的几何特征

        Returns:
            scalar 梯度惩罚项。batch 不等长时按较小的一方截断（异常样本可能
            按 ``num_anomalies_per_sample`` 成倍生成）。
        """
        batch_size = min(real_clip.shape[0], fake_clip.shape[0])
        if batch_size == 0:
            return real_clip.new_zeros(())

        real_clip = real_clip[:batch_size].detach()
        real_geo = real_geo[:batch_size].detach().to(real_clip.dtype)
        fake_clip = fake_clip[:batch_size].detach().to(real_clip.dtype)
        fake_geo = fake_geo[:batch_size].detach().to(real_clip.dtype)

        alpha = torch.rand(batch_size, 1, device=real_clip.device, dtype=real_clip.dtype)
        inter_clip = (alpha * real_clip + (1.0 - alpha) * fake_clip).requires_grad_(True)
        inter_geo = (alpha * real_geo + (1.0 - alpha) * fake_geo).requires_grad_(True)

        logits = self.forward(inter_clip, inter_geo, return_logits=True)
        gradients: Tuple[Tensor, ...] = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=[inter_clip, inter_geo],
            create_graph=True,
            only_inputs=True,
        )
        grad = torch.cat([g.flatten(1) for g in gradients], dim=1)
        return (grad.norm(2, dim=1) - 1.0).pow(2).mean()

    def extra_repr(self) -> str:
        return (
            f"clip_dim={self.clip_dim}, geo_dim={self.geo_dim}, "
            f"hidden_dim={self.hidden_dim}"
        )

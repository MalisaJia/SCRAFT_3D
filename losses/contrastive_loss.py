"""多视角对比学习损失（InfoNCE）。"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-8


class MultiViewContrastiveLoss(nn.Module):
    """多视角对比学习损失。

    正对：同一物体不同视角的语义特征（应相似）
    负对：不同物体的语义特征（应不同）

    使用 InfoNCE 风格的对比损失

    实现上采用 "负样本-only 分母" 的 multi-positive 形式：
    对每个 (anchor, 正样本) 对计算 -log(exp(pos/τ) / (exp(pos/τ) + Σexp(neg/τ)))，
    再对所有正对求平均。这样损失下界为 0（正对相似度远高于负对时），
    而不是把正样本也塞进分母导致下界变成 log(N_pos)，便于与其他损失加权组合。

    Args:
        temperature: 温度系数 τ，越小对困难负样本越敏感
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature 必须为正数，得到 {temperature}")
        self.temperature = temperature

    def forward(self, view_features: Tensor) -> Tensor:
        """
        Args:
            view_features: [B, N_views, D]

        实现：
        - 正对：同一B中同一物体的不同视角
        - 负对：不同物体的视角
        - InfoNCE: -log(exp(pos/τ) / Σexp(neg/τ))
        Returns:
            loss: scalar，下界 0（B < 2 或 N_views < 2 时返回 0）
        """
        if view_features.dim() != 3:
            raise ValueError(
                f"view_features 应为 [B, N_views, D]，得到 {tuple(view_features.shape)}"
            )
        batch, num_views, _ = view_features.shape
        if batch < 2 or num_views < 2:
            # 缺少正对或负对时无法构造 InfoNCE，返回 0
            return view_features.sum() * 0.0

        feats = F.normalize(view_features.flatten(0, 1), dim=-1, eps=EPS)
        total = feats.size(0)
        device = feats.device

        sim = feats @ feats.t() / self.temperature
        self_mask = torch.eye(total, dtype=torch.bool, device=device)

        # 物体归属：同一物体的所有视角共享 id
        obj_ids = torch.arange(batch, device=device).repeat_interleave(num_views)
        same_obj = obj_ids[:, None] == obj_ids[None, :]
        positive_mask = same_obj & ~self_mask  # 同物体不同视角
        negative_mask = ~same_obj  # 不同物体

        # 每个 anchor 在其所有负样本上的 logsumexp（数值稳定，屏蔽位填 -inf）
        neg_lse = torch.logsumexp(
            sim.masked_fill(~negative_mask, float("-inf")), dim=1, keepdim=True
        )

        # -log(exp(pos) / (exp(pos) + Σexp(neg))) == softplus(neg_lse - pos)
        pair_loss = F.softplus(neg_lse - sim)
        # 用 masked_fill 而非乘 mask，避免非正对位置的极端值参与求和
        pair_loss = pair_loss.masked_fill(~positive_mask, 0.0)
        num_pos = positive_mask.sum(dim=1).clamp(min=1)
        return (pair_loss.sum(dim=1) / num_pos).mean()

    def loss_dict(self, view_features: Tensor) -> Dict[str, Tensor]:
        """返回 dict 形式，便于与其他损失统一记录日志。"""
        loss = self.forward(view_features)
        return {"contrastive_loss": loss, "total": loss}

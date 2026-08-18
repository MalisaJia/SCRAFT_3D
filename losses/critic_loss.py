"""语义合理性判别损失模块。

配合 :class:`~models.semantic_critic.SemanticCritic` 使用：Critic 学习把
「正常」结构与 :class:`~utils.anomaly_constructor.AnomalyConstructor` 程序化
构造出的异常结构分开，Generator 则被推动生成让 Critic 判为 plausible 的结构。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["CriticLoss"]

EPS = 1e-7


class CriticLoss(nn.Module):
    """Semantic Critic 的训练损失。

    - 对 Critic：BCE loss，正样本(正常mesh)标签1，负样本(异常mesh)标签0
    - 对 Generator：鼓励生成让 Critic 判为 plausible 的结构

    Critic 损失 = BCE(critic(normal), 1) + BCE(critic(anomaly), 0)
    + gp_weight * gradient_penalty
    Generator 损失 = -log(critic(generated))  (或 BCE(critic(generated), 1))

    ``label_smoothing`` 采用 one-sided 形式（Salimans et al.）：只把正样本的
    目标从 1 降到 ``1 - label_smoothing``，负样本目标保持 0。双侧平滑会给
    异常样本一个「合理」的下限，反而削弱异常信号。

    Args:
        gp_weight: 梯度惩罚权重。
        label_smoothing: 正样本标签平滑幅度，取值 [0, 0.5)。
        from_logits: 输入是否为未激活的 logits。为 True 时走
            ``binary_cross_entropy_with_logits``（数值更稳定，推荐）；
            为 False 时输入被视为 [0, 1] 概率并做 clamp 后取 BCE。
    """

    def __init__(
        self,
        gp_weight: float = 10.0,
        label_smoothing: float = 0.1,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        if gp_weight < 0.0:
            raise ValueError(f"gp_weight 不能为负，得到 {gp_weight}")
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError(f"label_smoothing 应在 [0, 0.5)，得到 {label_smoothing}")

        self.gp_weight = gp_weight
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _bce(self, scores: Tensor, target: float) -> Tensor:
        """对 ``scores`` 施加常数目标的 BCE，返回标量。"""
        scores = scores.flatten()
        labels = torch.full_like(scores, target)
        if self.from_logits:
            return F.binary_cross_entropy_with_logits(scores, labels)
        return F.binary_cross_entropy(scores.clamp(EPS, 1.0 - EPS), labels)

    def _probability(self, scores: Tensor) -> Tensor:
        """把打分统一换算成 [0, 1] 概率（仅用于日志）。"""
        scores = scores.detach().flatten()
        return torch.sigmoid(scores) if self.from_logits else scores

    # ------------------------------------------------------------------ #
    # Critic
    # ------------------------------------------------------------------ #
    def critic_loss(
        self,
        normal_scores: Tensor,
        anomaly_scores: Tensor,
        gradient_penalty: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """计算 Critic 的训练损失。

        Args:
            normal_scores: [B, 1] 或 [B] Critic 对正常 mesh 的打分。
            anomaly_scores: [B', 1] 或 [B'] Critic 对异常 mesh 的打分
                （异常样本可按 ``num_anomalies_per_sample`` 成倍生成，
                与正常样本数量不必相等）。
            gradient_penalty: 可选的梯度惩罚标量，由
                ``SemanticCritic.compute_gradient_penalty`` 给出。

        Returns:
            dict with:
            - 'normal_loss' / 'anomaly_loss': 两路 BCE 子项
            - 'gradient_penalty': 加权前的梯度惩罚（未提供时为 0）
            - 'total': 加权求和后的总损失
            - 'normal_prob' / 'anomaly_prob': 平均 plausibility（无梯度，日志用）
            - 'accuracy': 以 0.5 为阈值的判别准确率（无梯度，日志用）
        """
        normal_loss = self._bce(normal_scores, 1.0 - self.label_smoothing)
        anomaly_loss = self._bce(anomaly_scores, 0.0)
        total = normal_loss + anomaly_loss

        if gradient_penalty is None:
            penalty = total.new_zeros(())
        else:
            penalty = gradient_penalty
            total = total + self.gp_weight * penalty

        normal_prob = self._probability(normal_scores)
        anomaly_prob = self._probability(anomaly_scores)
        correct = torch.cat([(normal_prob > 0.5).float(), (anomaly_prob <= 0.5).float()])

        return {
            "normal_loss": normal_loss,
            "anomaly_loss": anomaly_loss,
            "gradient_penalty": penalty,
            "total": total,
            "normal_prob": normal_prob.mean(),
            "anomaly_prob": anomaly_prob.mean(),
            "accuracy": correct.mean(),
        }

    # ------------------------------------------------------------------ #
    # Generator
    # ------------------------------------------------------------------ #
    def generator_loss(self, generated_scores: Tensor) -> Tensor:
        """计算 Generator 应承受的 critic 损失。

        即 BCE(critic(generated), 1) == -log(critic(generated))：让生成结构
        被 Critic 判为合理。这里**不做**标签平滑——生成器的目标应当是完全
        合理的结构，平滑后的目标会给它一个提前停止优化的借口。

        Args:
            generated_scores: [B, 1] 或 [B] Critic 对生成 mesh 的打分
                （需保留计算图，梯度经 CLIP / 几何特征回传到顶点）。

        Returns:
            scalar 损失。
        """
        return self._bce(generated_scores, 1.0)

    # ------------------------------------------------------------------ #
    # 统一入口
    # ------------------------------------------------------------------ #
    def forward(
        self,
        normal_scores: Tensor,
        anomaly_scores: Tensor,
        gradient_penalty: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """与其余损失模块保持一致的 ``forward``，等价于 :meth:`critic_loss`。"""
        return self.critic_loss(normal_scores, anomaly_scores, gradient_penalty)

    def extra_repr(self) -> str:
        return (
            f"gp_weight={self.gp_weight}, label_smoothing={self.label_smoothing}, "
            f"from_logits={self.from_logits}"
        )

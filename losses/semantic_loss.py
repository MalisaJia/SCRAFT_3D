"""多视角语义一致性损失（核心模块）。

利用 CLIP 特征在其联合嵌入空间中约束两件事：
1. 同一物体的不同视角特征应彼此一致（去掉视角带来的语义漂移）；
2. 每个视角的特征应与该视角对应的文本描述对齐
   （例如背面渲染应贴近 "a dog from the back"）。

作用边界：本损失只衡量「各视角看起来是否像同一个物体」以及「是否匹配
文本描述」这类物体级语义信号，惩罚的是视角间语义特征的不一致（这种不
一致可能暗示 Janus 多脸等几何异常），但它本身无法判断结构是否合理
（如腿的数量）。细粒度结构合理性判别由 ``models.semantic_critic.SemanticCritic``
通过有监督训练完成。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-8


class SemanticConsistencyLoss(nn.Module):
    """多视角语义一致性损失。

    核心思想：利用 CLIP 特征确保多视图间的语义一致性——同一物体从不同
    视角渲染后，其语义嵌入在 CLIP 空间中应该保持一致；惩罚视角间语义
    特征的不一致（可能暗示 Janus 问题等几何异常）。

    同时支持视角感知对齐：每个视角的语义特征
    应该与对应的视角感知文本描述对齐。

    Args:
        temperature: InfoNCE 温度系数，用于 ``view_text_alignment``
        consistency_weight: 多视角一致性项权重
        alignment_weight: 图文对齐项权重
        hardest_weight: 一致性项中「最不一致视角对」的额外惩罚权重
    """

    def __init__(
        self,
        temperature: float = 0.07,
        consistency_weight: float = 1.0,
        alignment_weight: float = 1.0,
        hardest_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature 必须为正数，得到 {temperature}")
        if hardest_weight < 0:
            raise ValueError(f"hardest_weight 不能为负，得到 {hardest_weight}")
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.alignment_weight = alignment_weight
        self.hardest_weight = hardest_weight

    def multi_view_consistency(self, view_features: Tensor) -> Tensor:
        """同一物体多视角特征的一致性损失（成对形式）。

        Args:
            view_features: [B, N_views, D] 各视角语义特征
        Returns:
            loss: scalar，越小表示多视角越一致

        实现：枚举所有视角对 (i, j), i < j，取余弦距离 ``1 - cos_sim``，
        损失 = 所有对的均值 + ``hardest_weight`` × 最大对的距离。

        动机：旧的「均值中心」形式先把各视角特征平均成语义中心，再算各视角
        到中心的距离，异常视角的偏离会被其余正常视角的均值稀释（例如只有某
        一个角度渲染出多脸的狗，均值仍接近正常语义，惩罚被摊薄）。成对形式
        让每两个视角直接比较，任意一对之间的不一致都会完整地体现在损失里；
        再显式加上最难视角对（hardest pair）项，把梯度集中到真正矛盾的那对
        视角上。
        """
        if view_features.dim() != 3:
            raise ValueError(
                f"view_features 应为 [B, N_views, D]，得到 {tuple(view_features.shape)}"
            )
        if view_features.size(1) < 2:
            # 单视角无一致性可言，返回 0（保留梯度图的 dtype/device）
            return view_features.sum() * 0.0

        feats = F.normalize(view_features, dim=-1, eps=EPS)
        # [B, N, N] 全部视角对的余弦相似度（特征已归一化，故矩阵乘即余弦）
        pairwise_sim = feats @ feats.transpose(-1, -2)
        # 余弦距离 = 1 - 余弦相似度，范围 [0, 2]
        pairwise_dist = 1.0 - pairwise_sim

        num_views = feats.size(1)
        # 只取上三角（i < j）：排除对角线自身对比，且每对只计一次
        pair_i, pair_j = torch.triu_indices(
            num_views, num_views, offset=1, device=feats.device
        )
        # [B, N*(N-1)/2]
        pair_dist = pairwise_dist[:, pair_i, pair_j]

        loss = pair_dist.mean()
        if self.hardest_weight > 0:
            # 每个物体各自最不一致的一对视角，再对 batch 求均值
            hardest = pair_dist.max(dim=1).values.mean()
            loss = loss + self.hardest_weight * hardest
        return loss

    def view_text_alignment(
        self, view_features: Tensor, text_features: Tensor
    ) -> Tensor:
        """视角感知的图像-文本对齐损失。

        Args:
            view_features: [B*N_views, D] 各视角语义特征
            text_features: [B*N_views, D] 对应的视角感知文本特征
        Returns:
            loss: scalar

        实现：对应视角的图像特征应与该视角的文本描述对齐
        例如：背面渲染的特征应与 "a dog from the back" 对齐
        """
        if view_features.dim() > 2:
            view_features = view_features.flatten(0, -2)
        if text_features.dim() > 2:
            text_features = text_features.flatten(0, -2)
        if view_features.shape != text_features.shape:
            raise ValueError(
                "view_features 与 text_features 形状需一致，得到 "
                f"{tuple(view_features.shape)} vs {tuple(text_features.shape)}"
            )

        img = F.normalize(view_features, dim=-1, eps=EPS)
        txt = F.normalize(text_features, dim=-1, eps=EPS)

        # 对称 InfoNCE：对角线为正确的 (视角, 视角感知文本) 配对
        logits = img @ txt.t() / self.temperature
        target = torch.arange(img.size(0), device=img.device)
        loss_i2t = F.cross_entropy(logits, target)
        loss_t2i = F.cross_entropy(logits.t(), target)
        return 0.5 * (loss_i2t + loss_t2i)

    def forward(
        self, view_features: Tensor, text_features: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Args:
            view_features: [B, N_views, D]，或已展平的 [B*N_views, D]
                （展平输入时仅当 ``text_features`` 给出才有意义）
            text_features: [B, N_views, D] 或 [B*N_views, D] 视角感知文本特征，
                为 None 时只计算一致性项
        Returns:
            dict with 'consistency_loss', 'alignment_loss', 'total_loss'
        """
        zero = view_features.sum() * 0.0

        if view_features.dim() == 3:
            consistency = self.multi_view_consistency(view_features)
            flat_views = view_features.flatten(0, 1)
        else:
            consistency = zero
            flat_views = view_features

        if text_features is None:
            alignment = zero
        else:
            flat_text = (
                text_features.flatten(0, 1)
                if text_features.dim() == 3
                else text_features
            )
            alignment = self.view_text_alignment(flat_views, flat_text)

        total = (
            self.consistency_weight * consistency + self.alignment_weight * alignment
        )
        return {
            "consistency_loss": consistency,
            "alignment_loss": alignment,
            "total_loss": total,
        }

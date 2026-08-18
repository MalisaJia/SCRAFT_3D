"""损失函数模块。

包含多视角语义一致性损失、多视角对比损失、mesh 几何正则化、
视角多样性损失以及语义合理性判别损失。

所有损失的 ``forward`` 均返回 ``Dict[str, Tensor]``（子项 + total），
便于加权组合与日志记录；``MultiViewContrastiveLoss`` 返回单个标量，
如需 dict 形式可调用其 ``loss_dict``。
"""

from .contrastive_loss import MultiViewContrastiveLoss
from .critic_loss import CriticLoss
from .geometry_reg import GeometryRegularization
from .semantic_loss import SemanticConsistencyLoss
from .view_diversity_loss import ViewDiversityLoss

__all__ = [
    "SemanticConsistencyLoss",
    "MultiViewContrastiveLoss",
    "GeometryRegularization",
    "ViewDiversityLoss",
    "CriticLoss",
]

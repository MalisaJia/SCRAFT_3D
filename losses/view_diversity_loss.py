"""视角多样性损失（核心创新之一）。"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-8


class ViewDiversityLoss(nn.Module):
    """视角多样性损失。

    核心创新：惩罚生成器重复渲染同一角度。

    思想：如果生成器学会只在某些"好"的角度看起来合理，
    那它可能在其他角度产生不合理的几何。
    通过最大化视角分布的熵，强制生成器在所有角度都产生合理的结果。

    实现方式：
    1. 统计训练过程中各方位角bin的使用频率
    2. 计算分布熵，熵越高越均匀
    3. 对低熵（集中在某些角度）施加惩罚

    另外，直接对语义特征施加视角均匀性约束：
    - 各视角的判别器打分不应有太大方差
    - 即模型在任何角度看都应该"像"目标对象

    实现细节：
    - 采用**软分箱**（soft binning）把角度分配到 bin，使损失对角度可微，
      从而也能训练可学习的相机采样器；角度不可微时该项退化为纯监控指标。
    - 用 EMA buffer 记录跨 step 的历史使用频率，单个 batch 视角数很少时
      仍能反映"训练过程中"的整体分布。
    - 熵损失以 ``log(K) - H`` 的归一化形式返回，取值 [0, 1]，
      0 表示完全均匀；与 ``-entropy`` 仅差一个常数，但数值范围更友好。

    Args:
        num_azimuth_bins: 方位角 bin 数
        num_elevation_bins: 仰角 bin 数
        elevation_range: 仰角取值范围（单位由 ``angle_unit`` 决定）。
            **必须与渲染器的实际采样范围一致**：若渲染器只在正负 30 度内采样，
            而这里按正负 90 度分箱，则所有样本只会落在中间几个 bin，
            熵亏损会留下无法优化的固定底噪。
        angle_unit: ``"radian"``（默认，对应 ``MultiViewRenderer.render()``
            返回的 ``azimuths``/``elevations``）或 ``"degree"``
            （可直接使用 config 中的 ``elevation_range: [-30, 30]``）。
        momentum: EMA 动量，越大越看重历史分布。需让有效记忆窗口
            ``1/(1-momentum)`` 乘以单步视角数达到 bin 总数量级，否则即使长期
            视角均匀，熵亏损也会留下固定底噪（8 bin、每步 4 个同角度视角时，
            momentum=0.5 底噪约 0.33，momentum=0.9 约 0.01）。
        entropy_weight: 熵损失权重
        variance_weight: 视角打分方差损失权重
        softness: 软分箱带宽相对 bin 宽度的比例，越小越接近硬分箱
    """

    def __init__(
        self,
        num_azimuth_bins: int = 8,
        num_elevation_bins: int = 4,
        elevation_range: Tuple[float, float] = (-math.pi / 2, math.pi / 2),
        angle_unit: str = "radian",
        momentum: float = 0.9,
        entropy_weight: float = 1.0,
        variance_weight: float = 1.0,
        softness: float = 0.5,
    ) -> None:
        super().__init__()
        if num_azimuth_bins < 2 or num_elevation_bins < 1:
            raise ValueError("num_azimuth_bins 至少 2，num_elevation_bins 至少 1")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum 应在 [0, 1) 内，得到 {momentum}")
        if angle_unit not in ("radian", "degree"):
            raise ValueError(
                f"angle_unit 应为 'radian' 或 'degree'，得到 {angle_unit!r}"
            )

        self.num_azimuth_bins = num_azimuth_bins
        self.num_elevation_bins = num_elevation_bins
        self.angle_unit = angle_unit
        # 内部统一用弧度计算
        if angle_unit == "degree":
            elevation_range = (
                math.radians(elevation_range[0]),
                math.radians(elevation_range[1]),
            )
        if elevation_range[1] <= elevation_range[0]:
            raise ValueError(f"elevation_range 需递增，得到 {elevation_range}")
        self.elevation_range = elevation_range
        self.momentum = momentum
        self.entropy_weight = entropy_weight
        self.variance_weight = variance_weight
        self.softness = softness

        # 跨 step 的历史使用频率（EMA），初始化为均匀分布
        self.register_buffer(
            "running_azimuth", torch.full((num_azimuth_bins,), 1.0 / num_azimuth_bins)
        )
        self.register_buffer(
            "running_elevation",
            torch.full((num_elevation_bins,), 1.0 / num_elevation_bins),
        )

    def reset_statistics(self) -> None:
        """把历史频率重置为均匀分布（如换数据集或阶段切换时调用）。"""
        self.running_azimuth.fill_(1.0 / self.num_azimuth_bins)
        self.running_elevation.fill_(1.0 / self.num_elevation_bins)

    def _to_radians(self, angles: Tensor) -> Tensor:
        """根据 ``angle_unit`` 把输入角度统一成弧度的浮点张量。"""
        if not angles.is_floating_point():
            # 容忍整型角度输入（如整度数），避免后续除法被截断
            angles = angles.float()
        if self.angle_unit == "degree":
            return angles * (math.pi / 180.0)
        return angles

    def _soft_histogram_azimuth(self, azimuths: Tensor) -> Tensor:
        """方位角软分箱直方图，返回 [num_azimuth_bins]，和为 1。"""
        flat = self._to_radians(azimuths).reshape(-1)
        bin_width = 2 * math.pi / self.num_azimuth_bins
        centers = torch.arange(
            self.num_azimuth_bins, device=flat.device, dtype=flat.dtype
        ) * bin_width

        # 圆周距离：把角度差折叠到 [-π, π]
        delta = flat[:, None] - centers[None, :]
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        sigma = max(self.softness, EPS) * bin_width
        weights = torch.softmax(-(delta / sigma).pow(2), dim=1)
        return weights.mean(dim=0)

    def _soft_histogram_elevation(self, elevations: Tensor) -> Tensor:
        """仰角软分箱直方图，返回 [num_elevation_bins]，和为 1。"""
        flat = self._to_radians(elevations).reshape(-1)
        if self.num_elevation_bins == 1:
            return torch.ones(1, device=flat.device, dtype=flat.dtype)

        low, high = self.elevation_range
        bin_width = (high - low) / self.num_elevation_bins
        centers = low + (
            torch.arange(
                self.num_elevation_bins, device=flat.device, dtype=flat.dtype
            )
            + 0.5
        ) * bin_width

        delta = flat[:, None] - centers[None, :]
        sigma = max(self.softness, EPS) * bin_width
        weights = torch.softmax(-(delta / sigma).pow(2), dim=1)
        return weights.mean(dim=0)

    @staticmethod
    def _entropy_deficit(hist: Tensor) -> Tensor:
        """归一化熵亏损 (log K - H) / log K，范围 [0, 1]，0 表示完全均匀。"""
        num_bins = hist.numel()
        if num_bins < 2:
            return hist.sum() * 0.0
        prob = hist / hist.sum().clamp(min=EPS)
        entropy = -(prob * torch.log(prob + EPS)).sum()
        max_entropy = math.log(num_bins)
        return ((max_entropy - entropy) / max_entropy).clamp(min=0.0)

    def angular_entropy_loss(self, azimuths: Tensor, elevations: Tensor) -> Tensor:
        """计算角度分布的熵损失。

        将方位角离散化到bins，计算使用频率分布的负熵
        目标：最大化熵（均匀分布）

        Args:
            azimuths: [B, N_views] 当前batch使用的方位角（单位同 ``angle_unit``）
            elevations: [B, N_views] 当前batch使用的仰角（单位同 ``angle_unit``）
        Returns:
            loss: -entropy（越小=越均匀=越好），以归一化熵亏损形式给出，
                  范围 [0, 1]
        """
        az_hist = self._soft_histogram_azimuth(azimuths)
        el_hist = self._soft_histogram_elevation(elevations)

        # 与历史 EMA 混合，衡量"训练过程中"的累计均匀性
        blended_az = (
            1.0 - self.momentum
        ) * az_hist + self.momentum * self.running_azimuth.to(az_hist.dtype)
        blended_el = (
            1.0 - self.momentum
        ) * el_hist + self.momentum * self.running_elevation.to(el_hist.dtype)

        if self.training:
            with torch.no_grad():
                self.running_azimuth.mul_(self.momentum).add_(
                    az_hist.detach().to(self.running_azimuth.dtype),
                    alpha=1.0 - self.momentum,
                )
                self.running_elevation.mul_(self.momentum).add_(
                    el_hist.detach().to(self.running_elevation.dtype),
                    alpha=1.0 - self.momentum,
                )

        az_loss = self._entropy_deficit(blended_az)
        if self.num_elevation_bins < 2:
            return az_loss
        el_loss = self._entropy_deficit(blended_el)
        return 0.5 * (az_loss + el_loss)

    def score_variance_loss(self, per_view_scores: Tensor) -> Tensor:
        """各视角判别分数的方差损失。

        如果模型在某些视角得分很高、其他视角得分低，
        说明模型不具有全方位的合理性。

        Args:
            per_view_scores: [B, N_views] 各视角的判别器打分或语义对齐分数
        Returns:
            loss: 分数的方差（越小=越一致=越好）
        """
        if per_view_scores.dim() == 1:
            per_view_scores = per_view_scores.unsqueeze(0)
        if per_view_scores.size(1) < 2:
            return per_view_scores.sum() * 0.0
        # 每个物体内部跨视角的方差，再对 batch 取平均
        return per_view_scores.var(dim=1, unbiased=False).mean()

    def forward(
        self,
        azimuths: Tensor,
        elevations: Tensor,
        per_view_scores: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            azimuths: [B, N_views] 方位角（单位同 ``angle_unit``）
            elevations: [B, N_views] 仰角（单位同 ``angle_unit``）
            per_view_scores: [B, N_views] 各视角打分，可选
        Returns:
            dict with 'entropy_loss', 'variance_loss' (if scores provided), 'total'
        """
        if azimuths.shape != elevations.shape:
            raise ValueError(
                "azimuths 与 elevations 形状需一致，得到 "
                f"{tuple(azimuths.shape)} vs {tuple(elevations.shape)}"
            )

        entropy_loss = self.angular_entropy_loss(azimuths, elevations)
        total = self.entropy_weight * entropy_loss
        out = {"entropy_loss": entropy_loss}

        if per_view_scores is not None:
            variance_loss = self.score_variance_loss(per_view_scores)
            total = total + self.variance_weight * variance_loss
            out["variance_loss"] = variance_loss

        out["total"] = total
        return out

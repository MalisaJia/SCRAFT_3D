"""冻结 CLIP 编码器封装。

CLIP 参数全程冻结，只作为语义先验使用；但图像分支保留计算图，
使 CLIP 语义损失的梯度能够穿过 CLIP 回传到渲染图像 -> mesh 顶点。

能力边界说明：
CLIP（尤其 ViT-B/32）的语义判别粒度为物体级别（「像不像狗」），
而非结构级别（「四条腿 vs 八条腿」）。本模块提供的是高质量的语义
特征表示与图文对齐信号；细粒度结构合理性判别由
:class:`~models.semantic_critic.SemanticCritic` 在该特征空间中
通过有监督训练（正常结构 vs 程序化构造的异常结构）实现。
"""

import random
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

# torchvision 的 functional 变换（resize / perspective）基于 autograd 算子实现，
# 对输入图像可微，可用于训练期的可微分渲染图增强；缺失时自动禁用增强
try:
    from torchvision.transforms import RandomPerspective
    from torchvision.transforms import functional as TF

    _TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    RandomPerspective = None
    TF = None
    _TORCHVISION_AVAILABLE = False

# 优先使用 OpenAI 官方 clip，缺失时回退到 open_clip
_CLIP_BACKEND: Optional[str] = None
try:
    import clip as _clip

    _CLIP_BACKEND = "clip"
except ImportError:  # pragma: no cover - 取决于运行环境
    _clip = None
    try:
        import open_clip as _open_clip

        _CLIP_BACKEND = "open_clip"
    except ImportError:
        _open_clip = None

# CLIP 预处理使用的通道均值 / 方差
CLIP_MEAN: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)

# 常用 CLIP backbone 的共享嵌入维度（键统一为 open_clip 的连字符写法）。
# 只用于「不加载权重」的配置预校验；实际维度始终以加载后的 embed_dim 为准。
CLIP_EMBED_DIMS: Dict[str, int] = {
    "ViT-B-32": 512,
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-L-14-336": 768,
    "EVA02-L-14": 768,
    "ViT-H-14": 1024,
    "ViT-bigG-14": 1280,
}

# open_clip 用连字符命名，OpenAI 官方 clip 用斜杠命名。这里只登记官方 clip
# 真实提供的 ViT 系列；其余名字（EVA02-L-14、ViT-bigG-14 等仅 open_clip 有）
# 原样透传，由后端自己报「模型不存在」。
_OPENAI_NAME_ALIASES: Dict[str, str] = {
    "ViT-B-32": "ViT-B/32",
    "ViT-B-16": "ViT-B/16",
    "ViT-L-14": "ViT-L/14",
    "ViT-L-14-336": "ViT-L/14@336px",
}


def _canonical_model_name(model_name: str) -> str:
    """统一成连字符写法（open_clip 风格），作为查表用的规范键。"""
    return model_name.replace("/", "-").replace("@336px", "-336")


class CLIPEncoder:
    """冻结的 CLIP 编码器，用于提取图像和文本的语义特征。

    职责范围：多视图语义一致性（各视角看起来是否像同一个物体）与
    图文对齐（渲染结果是否匹配文本描述）。判别粒度为物体级别，不承担
    「结构是否合理」这类细粒度判断——后者见 ``SemanticCritic``。
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        input_range: str = "zero_one",
        pretrained: str = "openai",
        expected_embed_dim: Optional[int] = None,
        augmentation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model_name: CLIP 模型名，如 "ViT-B/32"、"ViT-L/14"。斜杠与连字符
                两种写法等价，内部会按后端要求自动转换。
            device: 模型所在设备。
            input_range: 输入图像的数值范围，"zero_one" 表示 [0, 1]，
                "minus_one_one" 表示 [-1, 1]。
            pretrained: 仅 open_clip 后端使用的预训练权重标签。
            expected_embed_dim: 配置中声明的 clip_dim。给出时会与实际加载模型的
                embed_dim 比对，不一致直接报错，避免维度错配到下游才暴露。
            augmentation: 训练期渲染图增强配置（对应 config 的 vlm.augmentation 段），
                支持 enabled / crop_scale / perspective_distortion 三个字段；
                None 或 enabled=False 时 encode_images 不做任何增强。

        Raises:
            ValueError: expected_embed_dim 与实际 embed_dim 不一致。
        """
        if _CLIP_BACKEND is None:
            raise RuntimeError(
                "未检测到 CLIP，请安装其中之一："
                "pip install git+https://github.com/openai/CLIP.git 或 pip install open_clip_torch"
            )
        if input_range not in ("zero_one", "minus_one_one"):
            raise ValueError(f"未知的 input_range: {input_range}")

        self.model_name = model_name
        self.device = torch.device(device)
        self.input_range = input_range
        self.pretrained = pretrained
        self.backend = _CLIP_BACKEND

        # ---- 训练期渲染图增强（可微分随机裁剪 + 小透视畸变）----
        self.augmentation_cfg = dict(augmentation or {})
        self.augmentation_enabled = bool(self.augmentation_cfg.get("enabled", False))
        if self.augmentation_enabled and not _TORCHVISION_AVAILABLE:
            print(
                "[警告] 未安装 torchvision，vlm.augmentation 已自动禁用"
                "（pip install torchvision 后可启用）"
            )
            self.augmentation_enabled = False
        # 训练 / 推理模式开关：仅训练模式下做随机增强，由 train.py 在构建后打开
        self.training_mode = False

        if self.backend == "clip":
            # OpenAI 官方 clip 只认斜杠写法（ViT-B/32）
            self.model, _ = _clip.load(
                self.normalize_model_name(model_name, "clip"),
                device=self.device,
                jit=False,
            )
            self.tokenizer = _clip.tokenize
        else:  # pragma: no cover - 取决于运行环境
            open_clip_name = self.normalize_model_name(model_name, "open_clip")
            self.model, _, _ = _open_clip.create_model_and_transforms(
                open_clip_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = _open_clip.get_tokenizer(open_clip_name)

        # 冻结全部参数并切换到 eval，避免 BN / dropout 影响与梯度更新
        self.model = self.model.float().eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.register_buffers()

        # 声明维度与实际维度必须一致：判别器语义头、SemanticCritic 的输入维度
        # 都按 clip_dim 构建，错配会在前向时才炸，这里提前拦住。
        if expected_embed_dim is not None and int(expected_embed_dim) != self.embed_dim:
            raise ValueError(
                f"配置声明的 clip_dim={int(expected_embed_dim)} 与实际加载的 "
                f"{model_name} embed_dim={self.embed_dim} 不一致。请把 vlm.clip_dim、"
                f"discriminator.semantic_head_dim、semantic_critic.clip_dim 三者都改为 "
                f"{self.embed_dim}"
            )

    # ------------------------------------------------------------------ #
    # 模型名 / 维度工具（不加载权重，可用于配置预校验）
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize_model_name(model_name: str, backend: str) -> str:
        """把模型名转换成指定后端认识的写法。

        open_clip 用连字符（``ViT-B-32``），OpenAI 官方 clip 用斜杠
        （``ViT-B/32``），配置里两种写法都允许。

        Args:
            model_name: 配置中书写的模型名。
            backend: "clip" 或 "open_clip"。

        Returns:
            目标后端可直接使用的模型名。无法识别的名字（如 EVA02-L-14、RN50）
            原样返回，交由后端自己报错。
        """
        if backend == "open_clip":
            return _canonical_model_name(model_name)
        return _OPENAI_NAME_ALIASES.get(_canonical_model_name(model_name), model_name)

    @staticmethod
    def get_embed_dim(model_name: str, pretrained: str = "openai") -> Optional[int]:
        """获取指定模型的 embedding 维度（不加载权重）。

        用于启动前的配置验证。常用模型维度：

        - ViT-B/32、ViT-B/16: 512
        - ViT-L/14、EVA02-L-14: 768
        - ViT-H/14: 1024
        - ViT-bigG/14: 1280

        Args:
            model_name: 模型名，斜杠或连字符写法均可。
            pretrained: 预训练权重标签。当前所有已登记模型的维度都与权重来源
                无关，保留该参数以便后续按权重区分。

        Returns:
            已登记模型的维度；未登记时返回 None（此时只能加载后用
            :attr:`embed_dim` 读取实际维度）。
        """
        del pretrained  # 目前维度只由结构决定
        return CLIP_EMBED_DIMS.get(_canonical_model_name(model_name))

    def register_buffers(self) -> None:
        """预先构造归一化所需的 mean / std 张量。"""
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    @property
    def input_resolution(self) -> int:
        """CLIP 视觉塔要求的输入分辨率。"""
        visual = self.model.visual
        resolution = getattr(visual, "input_resolution", None)
        if resolution is None:  # open_clip 的属性名不同
            resolution = getattr(visual, "image_size", 224)
        if isinstance(resolution, (tuple, list)):
            resolution = resolution[0]
        return int(resolution)

    @property
    def embed_dim(self) -> int:
        """CLIP 共享嵌入空间的特征维度。"""
        projection = getattr(self.model, "text_projection", None)
        if isinstance(projection, torch.Tensor):
            return int(projection.shape[-1])
        if isinstance(projection, torch.nn.Linear):  # 新版 open_clip 用 Linear
            return int(projection.out_features)
        # 最后回退：用一次空文本编码推断维度
        return int(self.encode_text([""]).shape[-1])

    # ------------------------------------------------------------------ #
    # 图像 / 文本编码
    # ------------------------------------------------------------------ #
    def set_training_mode(self, mode: bool) -> None:
        """切换训练 / 推理模式：只有训练模式下 encode_images 才做随机增强。"""
        self.training_mode = bool(mode)

    def augment_images(self, images: Tensor) -> Tensor:
        """对渲染图做可微分随机增强（仅训练模式且启用增强时生效）。

        增强内容：
        1. 随机裁剪（scale 范围 ``crop_scale``，默认 0.7-1.0）后 resize 回原尺寸；
        2. 小幅度随机透视变换（``perspective_distortion``，建议 0.05-0.15，
           过大会破坏物体语义）。

        全部使用 ``torchvision.transforms.functional`` 的张量算子实现
        （crop/resize 走 interpolate、perspective 走 grid_sample），
        因此对输入图像完全可微：CLIP 语义梯度可穿过增强回传到渲染图像。

        Args:
            images: [B, 3, H, W] 渲染图像（归一化前，范围由 input_range 指定）。

        Returns:
            [B, 3, H, W] 增强后的图像；未启用或非训练模式时原样返回。
        """
        if not (self.augmentation_enabled and self.training_mode):
            return images
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"images 形状应为 [B, 3, H, W]，实际为 {tuple(images.shape)}")

        scale_range = self.augmentation_cfg.get("crop_scale", (0.7, 1.0))
        scale_low = float(scale_range[0])
        scale_high = float(scale_range[1])
        distortion = float(self.augmentation_cfg.get("perspective_distortion", 0.1))

        _, _, height, width = images.shape
        bilinear = TF.InterpolationMode.BILINEAR

        augmented: List[Tensor] = []
        for image in images:  # 逐样本独立随机采样，增强多样性
            # ---- 随机裁剪 + resize 回原尺寸（可微）----
            scale = random.uniform(scale_low, scale_high)
            crop_h = max(1, int(round(height * scale)))
            crop_w = max(1, int(round(width * scale)))
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            image = TF.resized_crop(
                image, top, left, crop_h, crop_w, [height, width],
                interpolation=bilinear,
            )
            # ---- 小幅度随机透视变换（grid_sample，可微）----
            if distortion > 0.0:
                startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
                params = RandomPerspective.get_params(width, height, distortion)
                # 不同 torchvision 版本的返回值不同：
                # 旧版返回四角偏移量 / 终点列表，新版返回 (startpoints, endpoints) 元组
                if isinstance(params, tuple) and len(params) == 2:
                    endpoints = list(params[1])
                else:
                    endpoints = list(params)
                image = TF.perspective(
                    image, startpoints, endpoints, interpolation=bilinear
                )
            augmented.append(image)
        return torch.stack(augmented, dim=0)

    def preprocess_images(self, images: Tensor) -> Tensor:
        """把渲染图像调整到 CLIP 输入规格（可微）。

        Args:
            images: [B, 3, H, W]，数值范围由 input_range 指定。

        Returns:
            [B, 3, R, R] 归一化后的图像张量。
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"images 形状应为 [B, 3, H, W]，实际为 {tuple(images.shape)}")

        images = images.to(self.device, dtype=torch.float32)
        if self.input_range == "minus_one_one":
            images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0)

        resolution = self.input_resolution
        if images.shape[-1] != resolution or images.shape[-2] != resolution:
            # 双线性插值保持可微，梯度可回传到渲染分辨率的图像
            images = F.interpolate(
                images,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (images - self._mean) / self._std

    def encode_images(
        self, images: Tensor, normalize: bool = True, augment: bool = True
    ) -> Tensor:
        """编码图像为 CLIP 特征 [B, D]。

        训练模式且启用 ``vlm.augmentation`` 时，先对图像做可微分随机增强
        （见 :meth:`augment_images`），再进入 CLIP 视觉塔；梯度可穿过增强、
        CLIP 一路回传到输入图像。

        Args:
            images: [B, 3, H, W] 归一化前的图像（范围见 input_range）。
            normalize: 是否对输出特征做 L2 归一化。
            augment: 是否允许随机增强。主损失路径保持 True；精修反馈等
                条件信号路径应传 False，保证训练 / 推理反馈分布一致。

        Returns:
            [B, D] 图像特征。CLIP 权重冻结，但梯度可回传到 images。
        """
        if augment:
            images = self.augment_images(images)
        features = self.model.encode_image(self.preprocess_images(images))
        features = features.float()
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, texts: List[str], normalize: bool = True) -> Tensor:
        """编码文本为 CLIP 特征 [B, D]。

        文本分支不需要梯度，使用 no_grad 以节省显存。
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens).float()
        return F.normalize(features, dim=-1) if normalize else features

    # ------------------------------------------------------------------ #
    # 相似度
    # ------------------------------------------------------------------ #
    def compute_similarity(
        self, image_features: Tensor, text_features: Tensor
    ) -> Tensor:
        """计算图像-文本余弦相似度 [B_img, B_text]。"""
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        return image_features @ text_features.t()

    def paired_similarity(
        self, image_features: Tensor, text_features: Tensor
    ) -> Tensor:
        """计算一一配对的余弦相似度 [B]，用于视角感知的逐视角语义损失。"""
        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError(
                "配对相似度要求图像与文本数量一致，"
                f"实际为 {image_features.shape[0]} 与 {text_features.shape[0]}"
            )
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        return (image_features * text_features).sum(dim=-1)

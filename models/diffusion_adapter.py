"""扩散形状模型适配器：把 TripoSG 包装成与本项目 mesh 契约兼容的生成器。

Semantic3D-GAN 的 GAN 管线以 ``{'vertices': [B, V, 3], 'faces': [F, 3]}`` 作为
统一的 mesh 契约（见 ``rendering/multi_view_render.py:MultiViewRenderer.render``
与 ``inference.py:export_obj``）。本模块把预训练的 TripoSG（image-conditioned
rectified-flow 扩散模型，输出 SDF 经八叉树 marching cubes 提取）封装在
``DiffusionMeshGenerator`` 之中，使下游渲染 / 评估 / LoRA 微调脚本无需感知
扩散模型内部细节即可复用同一套 mesh 契约。

真实 TripoSG API（VAST-AI-Research/TripoSG，diffusers 风格）：
- ``triposg.pipelines.pipeline_triposg.TripoSGPipeline`` —— 组装
  VAE (``TripoSGVAEModel``) + DiT (``TripoSGDiTModel``) + RectifiedFlow
  调度器 + DINOv2 图像编码器；``from_pretrained`` 直接吃 diffusers 目录
  布局权重（``vae/``、``transformer/``、``image_encoder_dinov2/`` ...）。
- 图像条件走 **DINOv2**（pipeline 内部 ``encode_image``），VAE 只负责
  形状 latent <-> SDF 的编解码，不做图像编码。
- ``pipeline(image=..., num_inference_steps=..., guidance_scale=...,
  dense_octree_depth=..., hierarchical_octree_depth=...,
  use_flash_decoder=...)`` 返回 ``TripoSGPipelineOutput(samples=[(verts,
  faces), ...], meshes=[trimesh.Trimesh, ...])``。

设计要点：
- **懒加载导入**：TripoSG 相关模块仅在实例化时导入，未安装 TripoSG 的
  环境下本仓库其余代码仍可正常导入 / 运行。
- **mesh 归一化**：输出 mesh 中心化到包围盒原点，并缩放至单位球内，与渲染器
  假设（camera_distance=2.5，物体位于原点）一致。
- **面环绕方向**：marching cubes 输出的面朝向可能不一致，统一修正为从外部
  观察时 CCW（逆时针）的外向环绕。
- **sm_120 稳健性**：统一走 ``use_flash_decoder=False`` 的 naive 八叉树提取
  路径（``triposg.inference_utils.hierarchical_extract_geometry``），不依赖
  需要即时编译的 CUDA flexicubes/flash 扩展，在 RTX 5090 (Blackwell) 上免编译。
- **文本条件**：TripoSG 本体是 image-conditioned 的，文本 prompt 通过
  text-to-image 钩子（可选）转为图像条件；无钩子时由种子派生一张确定性
  程序化条件图（不同种子 -> 不同条件图 -> 不同形状），配合 CLIP 种子重排序
  实现文本控制。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ["DiffusionMeshGenerator"]

# 无 T2I 钩子时兜底的 HF 仓库（中国大陆直连不稳定，建议配置本地 weights_path）
HF_REPO_ID = "VAST-AI/TripoSG"


class DiffusionMeshGenerator:
    """Wraps TripoSG diffusion model, outputs mesh contract compatible with MultiViewRenderer.

    输出契约（与 ``MeshGenerator.forward`` 对齐）::

        {
            "vertices": Tensor[1, V, 3]  # float32，已归一化到单位球内
            "faces":    Tensor[F, 3]     # int64，外向 CCW 环绕
        }

    该 dict 可直接传入 ``MultiViewRenderer.render(vertices, faces)`` 与
    ``inference.export_obj(path, vertices, faces)``。
    """

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Args:
            config: ``model.diffusion`` 配置段，支持键：
                - model_name: 模型名（默认 "TripoSG"）。
                - weights_path: diffusers 目录布局的权重目录（含 model_index.json
                  与 vae/ transformer/ 等子目录）；留空则尝试从 HuggingFace
                  ``VAST-AI/TripoSG`` 拉取。
                - num_steps: 扩散采样步数（默认 50）。
                - cfg_scale: classifier-free guidance 强度（默认 7.0）。
                - octree_resolution: SDF 八叉树分辨率（默认 256，映射为
                  ``hierarchical_octree_depth = log2(resolution)``）。
                - num_tokens: 形状 latent token 数（默认 2048，与预训练一致）。
                - dtype: 权重精度（"bf16"（默认，CUDA）/ "fp16" / "fp32"）。
                - lora_weights_path: 微调 LoRA 权重路径（train_diffusion.py 输出，
                  留空不注入）。
                - lora_target_modules: LoRA 注入的注意力投影名（默认
                  ["to_q", "to_v"]，需与训练侧配置一致）。
            device: 推理设备（nvdiffrast 渲染要求 CUDA；mesh 生成本身在 CPU 也可跑，
                但 bf16 权重会自动降级为 fp32）。

        Raises:
            RuntimeError: TripoSG 未安装或权重加载失败。
        """
        cfg = dict(config or {})
        self.device = torch.device(device)
        self.model_name = str(cfg.get("model_name", "TripoSG"))
        self.weights_path = str(cfg.get("weights_path", "") or "")
        self.num_steps = int(cfg.get("num_steps", 50))
        self.cfg_scale = float(cfg.get("cfg_scale", 7.0))
        self.octree_resolution = int(cfg.get("octree_resolution", 256))
        self.num_tokens = int(cfg.get("num_tokens", 2048))
        # 微调 LoRA 权重（train_diffusion.py 的输出），留空则不注入适配器
        self.lora_weights_path = str(cfg.get("lora_weights_path", "") or "")
        self.lora_target_modules = list(
            cfg.get("lora_target_modules", ["to_q", "to_v"])
        )

        # 文本 -> 图像钩子：fn(prompt: str) -> PIL.Image，由调用方按需注入
        self._t2i_hook: Optional[Callable[[str], Any]] = None
        # 惰性缓存，避免重复构建 CLIP（仅做种子重排序时才初始化）
        self._clip: Any = None

        # 权重精度：CUDA 默认 bf16（5090 原生支持），CPU 退回 fp32
        dtype_name = str(cfg.get("dtype", "") or "")
        if dtype_name:
            self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                          "fp32": torch.float32}.get(dtype_name, torch.bfloat16)
        else:
            self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # 懒加载组装 TripoSG pipeline（VAE + DiT + DINOv2 + 调度器，未安装时此处才抛错）
        self.pipeline = self._load_pipeline()
        self.vae = self.pipeline.vae
        self.dit = self.pipeline.transformer

        # 可选：给 DiT 注入 LoRA 适配器并加载微调权重
        self.lora_adapters: Dict[str, Any] = {}
        if self.lora_weights_path:
            self._load_lora_weights()

        # 推理阶段参数全程冻结
        for module in (self.vae, self.dit):
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # 懒加载：TripoSG pipeline
    # ------------------------------------------------------------------ #
    def _resolve_weights_dir(self) -> Optional[Path]:
        """解析 weights_path：文件 -> 其所在目录；目录 -> 本身；不存在 -> None。"""
        if not self.weights_path:
            return None
        path = Path(self.weights_path)
        if path.is_file():
            return path.parent
        if path.is_dir():
            return path
        return None

    def _load_pipeline(self):
        """组装 TripoSG pipeline：本地 diffusers 目录优先，否则 HuggingFace。"""
        try:
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        except ImportError as exc:
            raise RuntimeError(
                "未能导入 TripoSG（triposg.pipelines.pipeline_triposg）。"
                "请先克隆 VAST-AI-Research/TripoSG 仓库并安装其依赖"
                "（注意 RTX 5090 需要 torch>=2.7 cu128），"
                "或检查 PYTHONPATH 是否包含 TripoSG 仓库根目录。"
            ) from exc

        weights_dir = self._resolve_weights_dir()
        if weights_dir is not None:
            if not (weights_dir / "model_index.json").is_file():
                raise FileNotFoundError(
                    f"权重目录缺少 model_index.json（diffusers 目录布局）: {weights_dir}。"
                    "请下载 VAST-AI/TripoSG 完整仓库快照到该目录。"
                )
            pipeline = TripoSGPipeline.from_pretrained(str(weights_dir))
        else:
            # 未提供本地权重时尝试 HuggingFace 官方仓库
            pipeline = TripoSGPipeline.from_pretrained(HF_REPO_ID)
        return pipeline.to(self.device, self.dtype)

    def _load_lora_weights(self) -> None:
        """给 DiT 注入 LoRA 适配器并加载微调权重（``train_diffusion.py`` 的输出）。

        权重文件可以是 ``lora_weights.pt``（含 ``{'lora', 'ema'}``）或训练
        checkpoint（含 ``'lora'`` 键）；rank 从权重形状自动推断，目标模块
        取自 ``lora_target_modules`` 配置（需与训练侧一致，真实 DiT 的注意力
        投影名为 diffusers 风格的 ``to_q`` / ``to_v``）。

        Raises:
            FileNotFoundError: 权重文件不存在。
            KeyError: 权重中缺少 LoRA 键或模块名不匹配。
        """
        from train_diffusion import inject_lora, load_lora_state_dict

        path = Path(self.lora_weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 LoRA 权重: {path}")
        state = self._load_state(path)
        lora_state = state.get("lora", state) if isinstance(state, dict) else state

        # 从任一 lora_A 张量推断秩（lora_A 形状为 [rank, in_features]）
        rank = None
        for key, value in lora_state.items():
            if key.endswith(".lora_A"):
                rank = int(value.shape[0])
                break
        if rank is None:
            raise KeyError(f"LoRA 权重中未找到任何 *.lora_A 键: {path}")

        self.lora_adapters = inject_lora(
            self.dit, rank, self.lora_target_modules
        )
        load_lora_state_dict(self.lora_adapters, lora_state)
        print(
            f"[LoRA] 已加载微调权重: {path}（rank={rank}，"
            f"{len(self.lora_adapters)} 个适配层）"
        )

    @staticmethod
    def _load_state(path: Path) -> Dict[str, Tensor]:
        """加载 .safetensors / .pt 权重文件为 state_dict。"""
        suffix = path.suffix.lower()
        if suffix in (".safetensors",):
            from safetensors.torch import load_file

            return load_file(str(path))
        try:  # torch>=2.6 默认 weights_only=True
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - 老版本 torch
            return torch.load(str(path), map_location="cpu")

    # ------------------------------------------------------------------ #
    # 主接口
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: Optional[str] = None,
        image: Optional[Path] = None,
        seed: Optional[int] = None,
        num_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        octree_resolution: Optional[int] = None,
        num_candidates: int = 1,
        renderer: Optional[Any] = None,
    ) -> Dict[str, Tensor]:
        """Generate a mesh from text prompt or image.

        Args:
            prompt: 文本描述（与 image 至少提供一个）。TripoSG 为 image-conditioned，
                文本 prompt 会先经 text-to-image 钩子转为图像条件；无钩子时由
                seed 派生确定性程序化条件图（不同 seed -> 不同形状）。
            image: 条件图像路径；提供时优先于 prompt。
            seed: 随机种子；None 表示不固定。
            num_steps: 扩散采样步数，None 用配置默认值。
            cfg_scale: guidance 强度，None 用配置默认值。
            octree_resolution: SDF 提取分辨率，None 用配置默认值。
            num_candidates: 仅 prompt 路径生效 —— 采样多个种子生成候选 mesh，
                用 CLIP 图文相似度重排序取最优（>1 且 CLIP 可用时）。
            renderer: 可选 ``MultiViewRenderer``，种子重排序渲染候选时使用；
                未提供则内部按渲染器默认参数构建。

        Returns:
            Dict with 'vertices' [1,V,3] float tensor and 'faces' [F,3] long tensor.

        Raises:
            ValueError: prompt 与 image 均未提供。
        """
        if prompt is None and image is None:
            raise ValueError("prompt 与 image 至少需要提供一个")

        num_steps = int(self.num_steps if num_steps is None else num_steps)
        cfg_scale = float(self.cfg_scale if cfg_scale is None else cfg_scale)
        octree_resolution = int(
            self.octree_resolution if octree_resolution is None else octree_resolution
        )

        # 1. 确定图像条件
        condition_image = None
        if image is not None:
            condition_image = self._load_condition_image(image)
        elif self._t2i_hook is not None:
            # 外部注入的 text-to-image 钩子（如 Stable Diffusion）优先
            condition_image = self._t2i_hook(prompt)
        else:
            # 无 T2I：CLIP 种子重排序 —— 不同种子派生不同程序化条件图，
            # 产生形状差异，用 CLIP 对渲染结果打分挑选与 prompt 最匹配的候选
            if seed is None and num_candidates > 1 and self._ensure_clip() is not None:
                return self._clip_seed_rerank(
                    prompt, num_candidates, num_steps, cfg_scale, octree_resolution, renderer
                )
            # 退化为种子派生的程序化条件图（无外部引导）
            print("[警告] 无图像条件且 text-to-image 不可用，使用种子派生程序化条件图")

        # 2. 固定随机种子（可复现）
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        with torch.no_grad():
            # 3. 扩散采样 -> SDF -> 网格提取（条件图为 None 时用 seed 派生）
            vertices_np, faces_np = self._sample_mesh(
                condition_image, num_steps, cfg_scale, octree_resolution, generator, seed
            )

        # 4-6. 归一化 + 外向 CCW 环绕 + 转张量
        vertices, faces = self._postprocess(vertices_np, faces_np)
        return {"vertices": vertices, "faces": faces}

    def set_text_to_image_hook(self, hook: Callable[[str], Any]) -> None:
        """注入 text-to-image 钩子：``hook(prompt) -> PIL.Image``。

        TripoSG 是 image-conditioned 的，接入外部 T2I 模型（如 Stable Diffusion）
        后即可把文本 prompt 转成图像条件，效果优于程序化条件图 + CLIP 种子重排序。
        """
        self._t2i_hook = hook

    # ------------------------------------------------------------------ #
    # 文本条件辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_condition_image(image: Path):
        """加载条件图像为 PIL.Image（懒加载 PIL）。"""
        from PIL import Image

        image = Path(image)
        if not image.is_file():
            raise FileNotFoundError(f"条件图像不存在: {image}")
        return Image.open(image).convert("RGB")

    @staticmethod
    def _synthesize_condition_image(seed: int, size: int = 224):
        """由种子派生确定性程序化条件图（平滑随机色斑拼图）。

        TripoSG 的 DINOv2 编码器对任意 RGB 图都能给出嵌入；不同种子的图
        纹理布局不同，引导扩散产生不同形状，作为无 T2I 钩子时的兜底条件。
        """
        from PIL import Image

        rng = np.random.default_rng(int(seed))
        grid = 6
        coarse = rng.random((grid, grid, 3)).astype(np.float32)
        zoom = max(1, size // grid)
        field = np.kron(coarse, np.ones((zoom, zoom, 1), dtype=np.float32))
        yy, xx = np.mgrid[0 : field.shape[0], 0 : field.shape[1]]
        ramp = 0.15 * (xx / max(field.shape[1] - 1, 1)).astype(np.float32)[..., None]
        field = np.clip(field * 0.85 + ramp, 0.0, 1.0)
        field = field[:size, :size]
        return Image.fromarray((field * 255).astype(np.uint8), "RGB")

    def _ensure_clip(self) -> Optional[Any]:
        """惰性构建 CLIP 编码器（用于种子重排序），不可用时返回 None。"""
        if self._clip is not None:
            return self._clip
        try:
            from vlm.clip_encoder import CLIPEncoder

            self._clip = CLIPEncoder(
                model_name="ViT-B/32",
                device=str(self.device),
                input_range="zero_one",  # 渲染器输出已在 [0, 1]
            )
        except Exception as exc:  # pragma: no cover - 取决于运行环境
            print(f"[警告] CLIP 初始化失败（{exc}），跳过种子重排序")
            self._clip = None
        return self._clip

    def _clip_seed_rerank(
        self,
        prompt: str,
        num_candidates: int,
        num_steps: int,
        cfg_scale: float,
        octree_resolution: int,
        renderer: Optional[Any],
    ) -> Dict[str, Tensor]:
        """CLIP 种子重排序：采样多个随机种子，按图文相似度挑选最优 mesh。

        与 ``inference.search_latent`` 的思路一致，只是候选维度从 GAN 潜变量
        换成了扩散采样种子（种子同时决定程序化条件图与初始噪声）。
        """
        from rendering.multi_view_render import MultiViewRenderer

        clip = self._ensure_clip()
        if renderer is None:
            renderer = MultiViewRenderer(
                num_views=4, azimuth_strategy="fixed", device=str(self.device)
            )
        text_features = clip.encode_text([prompt])  # [1, D]，已 L2 归一化

        # 种子池：用全局 RNG 派生，保证整体可复现
        seeds = torch.randint(0, 2**31 - 1, (int(num_candidates),)).tolist()
        best: Optional[Dict[str, Tensor]] = None
        best_score = -float("inf")

        for i, cand_seed in enumerate(seeds):
            print(f"  [重排序] 候选 {i + 1}/{len(seeds)}  seed={cand_seed}")
            generator = torch.Generator(device=self.device).manual_seed(int(cand_seed))
            try:
                with torch.no_grad():
                    verts_np, faces_np = self._sample_mesh(
                        None, num_steps, cfg_scale, octree_resolution, generator,
                        cand_seed,
                    )
                mesh_vertices, mesh_faces = self._postprocess(verts_np, faces_np)
                mesh = {"vertices": mesh_vertices, "faces": mesh_faces}
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                print(f"  [警告] 候选生成失败（{exc}），跳过")
                continue

            with torch.no_grad():
                out = renderer.render(mesh["vertices"], mesh["faces"])
                feats = clip.encode_images(out["images"].flatten(0, 1))
                feats = feats.mean(dim=0, keepdim=True)
                score = float(clip.compute_similarity(feats, text_features).item())
            if score > best_score:
                best_score, best = score, mesh

        if best is None:
            raise RuntimeError("所有候选 mesh 生成均失败，无法完成种子重排序")
        print(f"  [重排序] 最优 CLIP 分数: {best_score:.4f}")
        return best

    # ------------------------------------------------------------------ #
    # 扩散采样与网格提取
    # ------------------------------------------------------------------ #
    @staticmethod
    def _octree_depth(resolution: int) -> int:
        """``octree_resolution`` -> 八叉树深度（res = 2^depth，截断到 [6, 10]）。"""
        depth = int(round(float(np.log2(max(int(resolution), 8)))))
        return min(max(depth, 6), 10)

    def _sample_mesh(
        self,
        image: Optional[Any],
        num_steps: int,
        cfg_scale: float,
        octree_resolution: int,
        generator: Optional[torch.Generator],
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """完整采样管线：图像条件 -> 扩散采样 -> SDF 解码 -> mesh 提取。

        统一走官方 ``TripoSGPipeline``（内部完成 DINOv2 编码、CFG、
        rectified-flow 积分与八叉树 SDF 解码），``use_flash_decoder=False``
        保证在 sm_120 上无需编译 CUDA 扩展。

        Returns:
            (vertices [V, 3] float32 numpy, faces [F, 3] int64 numpy)
        """
        if image is None:
            # 兜底条件：种子派生程序化图（种子缺省时随机取一个）
            fallback_seed = int(seed) if seed is not None else int(
                torch.randint(0, 2**31 - 1, (1,)).item()
            )
            image = self._synthesize_condition_image(fallback_seed)

        depth = self._octree_depth(octree_resolution)
        output = self.pipeline(
            image=image,
            num_inference_steps=num_steps,
            num_tokens=self.num_tokens,
            guidance_scale=cfg_scale,
            generator=generator,
            dense_octree_depth=max(depth - 2, 4),
            hierarchical_octree_depth=depth,
            use_flash_decoder=False,
        )

        # TripoSGPipelineOutput: samples=[(verts, faces), ...], meshes=[trimesh.Trimesh]
        samples = getattr(output, "samples", None)
        if samples:
            verts, tris = samples[0]
            return (
                np.asarray(verts, dtype=np.float32),
                np.asarray(tris, dtype=np.int64),
            )
        meshes = getattr(output, "meshes", None)
        if meshes:
            mesh = meshes[0]
            return (
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.int64),
            )
        raise RuntimeError("TripoSG 采样未产生任何 mesh（SDF 等值面为空）")

    # ------------------------------------------------------------------ #
    # 后处理：归一化 + 面环绕
    # ------------------------------------------------------------------ #
    def _postprocess(
        self, vertices_np: np.ndarray, faces_np: np.ndarray
    ) -> Tuple[Tensor, Tensor]:
        """numpy mesh -> 归一化 + 外向 CCW -> 契约张量。"""
        vertices, faces = self._canonicalize_mesh(vertices_np, faces_np)
        vertices, faces = self._ensure_winding(vertices, faces)
        return (
            torch.from_numpy(vertices).float().to(self.device).unsqueeze(0),  # [1, V, 3]
            torch.from_numpy(faces).long().to(self.device),  # [F, 3]
        )

    def _canonicalize_mesh(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Center on bounding box, scale to fit unit sphere.

        渲染器假设物体位于原点且相机距离为 2.5，因此把包围盒中心平移到原点、
        最大半径缩放到 1（单位球内），保证任意来源的 mesh 都能被一致地渲染。

        Args:
            vertices: [V, 3] numpy。
            faces: [F, 3] numpy（本步骤不修改，原样透传）。

        Returns:
            归一化后的 (vertices, faces) numpy 数组。
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)

        center = 0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
        vertices = vertices - center
        radius = float(np.linalg.norm(vertices, axis=1).max())
        if radius < 1e-8:
            raise RuntimeError("mesh 退化：所有顶点重合，无法归一化")
        vertices = vertices / radius  # 最大半径 = 1，整体落入单位球
        return vertices, faces

    def _ensure_winding(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Verify and fix face winding for outward normals (CCW).

        marching cubes 的面朝向约定不一，这里用启发式统一为外向 CCW：
        对每个面，若其几何法线背离质心（指向外侧）则保持顶点顺序，
        否则翻转该面 —— 从外部观察时即 CCW。对闭合 2-流形 mesh 该启发式
        是可靠的（TripoSG 的 SDF 等值面即闭合曲面）。

        Args:
            vertices: [V, 3] numpy。
            faces: [F, 3] numpy。

        Returns:
            修正环绕后的 (vertices, faces) numpy 数组。
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        centroids = (v0 + v1 + v2) / 3.0
        mesh_centroid = vertices.mean(axis=0, keepdims=True)

        # 法线与 "面中心 -> 外部" 方向同向才算外向；否则翻转该面
        outward = np.sum(face_normals * (centroids - mesh_centroid), axis=1) > 0
        flipped = faces.copy()
        flipped[~outward] = faces[~outward][:, ::-1]

        num_flipped = int((~outward).sum())
        if num_flipped > 0:
            print(f"[网格后处理] 翻转 {num_flipped}/{faces.shape[0]} 个面以保证外向 CCW 环绕")
        return vertices, flipped

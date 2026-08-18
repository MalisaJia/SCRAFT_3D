"""可视化工具集合。

提供三类功能：
1. mesh 渲染可视化：多视图拼图（依赖 nvdiffrast 渲染器）、线框图、法线图
2. 训练过程可视化：解析 ``train.py`` 写出的 ``train.log`` 并绘制 loss 曲线
3. 论文用对比图：把多个方法的结果横向排版

matplotlib 为可选依赖：缺失时退化为纯 PIL 实现（功能略简化但不报错）。
PIL 属于核心依赖（见 requirements.txt），保存图片必需。
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch 张量输入的支持（本项目一定有 torch，这里仍做保护）
    import torch
    from torch import Tensor

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    torch = None
    Tensor = Any  # type: ignore[misc, assignment]
    _TORCH_AVAILABLE = False

try:
    from PIL import Image, ImageDraw

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    Image = None
    ImageDraw = None
    _PIL_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # 无显示环境（服务器 / SSH）下也能保存图片
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    matplotlib = None
    plt = None
    _MATPLOTLIB_AVAILABLE = False

__all__ = [
    "to_uint8_images",
    "tile_images",
    "save_image",
    "render_multiview_grid",
    "plot_training_curves",
    "visualize_normals",
    "render_wireframe",
    "create_comparison_figure",
]

# 曲线 / 对比图使用的调色板（RGB）
_PALETTE: Sequence[Tuple[int, int, int]] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


# ====================================================================== #
# 基础工具：数组格式转换 / 拼图 / 保存
# ====================================================================== #
def _to_numpy(array: Any) -> np.ndarray:
    """把 torch 张量或类数组对象转成 numpy 数组（自动 detach + 转 CPU）。"""
    if _TORCH_AVAILABLE and isinstance(array, torch.Tensor):
        return array.detach().float().cpu().numpy()
    return np.asarray(array)


def to_uint8_images(images: Any) -> np.ndarray:
    """把任意常见图像张量格式统一成 ``[N, H, W, 3]`` 的 uint8 数组。

    支持的输入形状：``[H, W]``、``[H, W, 3]``、``[3, H, W]``、
    ``[N, H, W, 3]``、``[N, 3, H, W]``、``[B, N, 3, H, W]``（前两维会展平）。
    浮点输入按最大值判断范围：``<= 1`` 视为 [0, 1]，否则视为 [0, 255]。

    Args:
        images: 图像数组 / 张量。

    Returns:
        [N, H, W, 3] uint8 数组。
    """
    array = _to_numpy(images)
    if array.ndim == 5:  # [B, N, C, H, W] -> [B*N, C, H, W]
        array = array.reshape(-1, *array.shape[2:])
    if array.ndim == 2:  # [H, W] 灰度
        array = array[None, ..., None]
    elif array.ndim == 3:
        # 通道优先还是通道在后：靠首/末维是否为 1 / 3 判断
        if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = array.transpose(1, 2, 0)[None]
        else:
            array = array[None]
    elif array.ndim == 4:
        if array.shape[1] in (1, 3) and array.shape[-1] not in (1, 3):
            array = array.transpose(0, 2, 3, 1)
    else:
        raise ValueError(f"无法解析的图像形状: {array.shape}")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:  # 丢弃 alpha 通道
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise ValueError(f"图像通道数应为 1 / 3 / 4，实际为 {array.shape[-1]}")

    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        array = array * scale
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def tile_images(
    images: Any,
    ncols: Optional[int] = None,
    padding: int = 2,
    background: int = 255,
) -> np.ndarray:
    """把一组图像拼成网格图。

    Args:
        images: 任意 ``to_uint8_images`` 可解析的图像集合。
        ncols: 网格列数，默认取 ``ceil(sqrt(N))``（接近正方形）。
        padding: 图像之间的间隔像素。
        background: 间隔区域的灰度值（0-255）。

    Returns:
        [H, W, 3] uint8 拼图。
    """
    frames = to_uint8_images(images)
    num, height, width, _ = frames.shape
    if num == 0:
        raise ValueError("images 为空，无法拼图")

    ncols = max(1, int(ncols) if ncols else int(math.ceil(math.sqrt(num))))
    nrows = int(math.ceil(num / ncols))

    canvas_h = nrows * height + (nrows + 1) * padding
    canvas_w = ncols * width + (ncols + 1) * padding
    canvas = np.full((canvas_h, canvas_w, 3), background, dtype=np.uint8)

    for index in range(num):
        row, col = divmod(index, ncols)
        top = padding + row * (height + padding)
        left = padding + col * (width + padding)
        canvas[top : top + height, left : left + width] = frames[index]
    return canvas


def save_image(image: Any, path: str) -> str:
    """保存图像到磁盘（自动创建父目录）。

    Args:
        image: 图像数组 / 张量 / PIL Image。
        path: 输出路径，扩展名决定格式。

    Returns:
        实际写入的路径。
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("保存图片需要 Pillow，请先 pip install Pillow")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(path)
        return path
    frames = to_uint8_images(image)
    array = frames[0] if frames.shape[0] == 1 else tile_images(frames)
    Image.fromarray(array).save(path)
    return path


# ====================================================================== #
# 1. 多视图渲染拼图
# ====================================================================== #
def render_multiview_grid(
    renderer: Any,
    vertices: Any,
    faces: Any,
    num_views: int = 8,
    image_size: int = 256,
    vertex_colors: Any = None,
    ncols: Optional[int] = None,
    padding: int = 2,
    return_pil: bool = True,
) -> Any:
    """渲染 mesh 的多视图并拼成网格图。

    临时覆盖 ``renderer.num_views`` / ``renderer.image_size``，函数退出时恢复，
    因此可以安全地复用训练时构建的渲染器实例。

    Args:
        renderer: ``MultiViewRenderer`` 实例。
        vertices: [V, 3] 或 [B, V, 3]（B > 1 时只渲染第 0 个样本）。
        faces: [F, 3] 或 [B, F, 3]。
        num_views: 渲染视角数量。
        image_size: 单视图分辨率。
        vertex_colors: 可选 [V, 3] 顶点颜色，范围 [0, 1]。
        ncols: 拼图列数，默认接近正方形。
        padding: 视图之间的间隔像素。
        return_pil: True 返回 PIL Image，False 返回 [H, W, 3] uint8 数组。

    Returns:
        PIL Image 或 numpy 数组。
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("render_multiview_grid 需要 torch")

    verts = vertices if isinstance(vertices, torch.Tensor) else torch.as_tensor(vertices)
    tris = faces if isinstance(faces, torch.Tensor) else torch.as_tensor(faces)
    verts = verts.detach().float()
    tris = tris.detach().long()

    if verts.dim() == 2:
        verts = verts.unsqueeze(0)
    verts = verts[:1]  # 只取第一个样本，避免拼图混入不同形状
    if tris.dim() == 3:
        tris = tris[0]

    colors = None
    if vertex_colors is not None:
        colors = (
            vertex_colors
            if isinstance(vertex_colors, torch.Tensor)
            else torch.as_tensor(vertex_colors)
        ).detach().float()
        if colors.dim() == 2:
            colors = colors.unsqueeze(0)
        colors = colors[:1].to(verts.device)

    old_views, old_size = renderer.num_views, renderer.image_size
    try:
        renderer.num_views = int(num_views)
        renderer.image_size = int(image_size)
        with torch.no_grad():
            out = renderer.render(verts, tris, vertex_colors=colors)
        images = out["images"][0]  # [N, 3, H, W]
    finally:
        renderer.num_views, renderer.image_size = old_views, old_size

    grid = tile_images(images, ncols=ncols, padding=padding)
    if return_pil:
        if not _PIL_AVAILABLE:
            raise RuntimeError("return_pil=True 需要 Pillow，请先 pip install Pillow")
        return Image.fromarray(grid)
    return grid


# ====================================================================== #
# 2. 训练曲线
# ====================================================================== #
# train.py 的日志形如：
#   [12:34:56] iter 100/100000  (0.52s/it)  views=4  w_sem=0.100
#       d/adv=0.6931  d/real_logit=0.1234  g/total=3.2100
_ITER_PATTERN = re.compile(r"\biter\s+(\d+)\s*/\s*\d+")
_METRIC_PATTERN = re.compile(r"([A-Za-z][\w/]*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
# views / batch_size 是配置项而非指标；s/it 不符合 key=value 形式，天然不会被匹配
_SKIP_KEYS = {"views", "batch_size"}


def parse_training_log(log_file: str) -> Dict[str, Tuple[List[int], List[float]]]:
    """解析训练日志，抽取每个指标的 (迭代数, 数值) 序列。

    Args:
        log_file: ``train.py`` 输出的日志文件（``<output_dir>/train.log``）。

    Returns:
        ``{metric_name: (iterations, values)}``。
    """
    if not os.path.isfile(log_file):
        raise FileNotFoundError(f"日志文件不存在: {log_file}")

    curves: Dict[str, Tuple[List[int], List[float]]] = {}
    current_iter: Optional[int] = None
    with open(log_file, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = _ITER_PATTERN.search(line)
            if match:
                current_iter = int(match.group(1))
            if current_iter is None:
                continue  # 首个 iter 行之前是启动信息，其中的 key=value 不是指标
            for key, raw in _METRIC_PATTERN.findall(line):
                if key in _SKIP_KEYS:
                    continue
                try:
                    value = float(raw)
                except ValueError:  # pragma: no cover - 正则已保证可转换
                    continue
                iterations, values = curves.setdefault(key, ([], []))
                iterations.append(current_iter)
                values.append(value)
    return curves


def _smooth(values: Sequence[float], window: int) -> List[float]:
    """滑动平均平滑（window <= 1 时原样返回）。"""
    if window <= 1 or len(values) <= 1:
        return list(values)
    window = min(int(window), len(values))
    out: List[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        out.append(running / min(index + 1, window))
    return out


def plot_training_curves(
    log_file: str,
    output_path: str,
    keys: Optional[Sequence[str]] = None,
    smooth: int = 1,
) -> str:
    """从训练日志文件解析 loss 并绘制曲线。

    指标按 ``d/`` ``g/`` 等前缀自动分组，每组一个子图（matplotlib 可用时）。

    Args:
        log_file: 训练日志路径。
        output_path: 输出图片路径。
        keys: 只绘制指定指标；None 表示全部。
        smooth: 滑动平均窗口，1 表示不平滑。

    Returns:
        输出图片路径。

    Raises:
        FileNotFoundError: 日志文件不存在。
        ValueError: 日志中没有解析到任何指标。
    """
    curves = parse_training_log(log_file)
    if keys is not None:
        wanted = set(keys)
        curves = {k: v for k, v in curves.items() if k in wanted}
    if not curves:
        raise ValueError(f"未能从 {log_file} 解析到任何指标，请确认日志格式")

    smoothed = {
        name: (iterations, _smooth(values, smooth))
        for name, (iterations, values) in sorted(curves.items())
    }

    if not _MATPLOTLIB_AVAILABLE:
        return _pil_line_chart(smoothed, output_path, title="training curves")

    # 按 "前缀/名称" 的前缀分组，没有前缀的归入 misc
    groups: Dict[str, List[str]] = {}
    for name in smoothed:
        prefix = name.split("/")[0] if "/" in name else "misc"
        groups.setdefault(prefix, []).append(name)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    num_groups = len(groups)
    fig, axes = plt.subplots(
        num_groups, 1, figsize=(10, 3.2 * num_groups), squeeze=False
    )
    for axis, (prefix, names) in zip(axes[:, 0], sorted(groups.items())):
        for index, name in enumerate(names):
            iterations, values = smoothed[name]
            color = np.asarray(_PALETTE[index % len(_PALETTE)]) / 255.0
            axis.plot(iterations, values, label=name, color=color, linewidth=1.2)
        axis.set_title(f"{prefix} metrics")
        axis.set_xlabel("iteration")
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _pil_line_chart(
    curves: Dict[str, Tuple[List[int], List[float]]],
    output_path: str,
    title: Optional[str] = None,
    size: Tuple[int, int] = (1000, 600),
) -> str:
    """matplotlib 缺失时的折线图兜底实现（每条曲线各自归一化到 [0, 1]）。"""
    if not _PIL_AVAILABLE:
        raise RuntimeError("绘制曲线需要 matplotlib 或 Pillow，请至少安装其中之一")

    width, height = size
    left, right, top, bottom = 70, 240, 40, 50
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)

    canvas = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(0, 0, 0))
    if title:
        draw.text((left, 12), title, fill=(0, 0, 0))

    max_iter = max(
        (max(iterations) for iterations, _ in curves.values() if iterations), default=1
    )
    max_iter = max(max_iter, 1)
    draw.text((left, top + plot_h + 12), "0", fill=(0, 0, 0))
    draw.text((left + plot_w - 40, top + plot_h + 12), str(max_iter), fill=(0, 0, 0))
    draw.text((8, top + plot_h // 2), "normalized", fill=(0, 0, 0))

    for index, (name, (iterations, values)) in enumerate(sorted(curves.items())):
        if not values:
            continue
        color = _PALETTE[index % len(_PALETTE)]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        points = [
            (
                left + plot_w * (it / max_iter),
                top + plot_h * (1.0 - (value - low) / span),
            )
            for it, value in zip(iterations, values)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        else:
            draw.ellipse(
                [points[0][0] - 2, points[0][1] - 2, points[0][0] + 2, points[0][1] + 2],
                fill=color,
            )
        legend_y = top + 4 + index * 14
        if legend_y < top + plot_h - 10:
            draw.line(
                [left + plot_w + 10, legend_y + 4, left + plot_w + 34, legend_y + 4],
                fill=color,
                width=3,
            )
            draw.text(
                (left + plot_w + 40, legend_y),
                f"{name} [{low:.3g}, {high:.3g}]",
                fill=(0, 0, 0),
            )
    return save_image(canvas, output_path)


# ====================================================================== #
# 3. 几何可视化（法线 / 线框）
# ====================================================================== #
def _mesh_arrays(vertices: Any, faces: Any) -> Tuple[np.ndarray, np.ndarray]:
    """把 mesh 转成 numpy 数组，并去掉可能存在的 batch 维。"""
    verts = _to_numpy(vertices).astype(np.float32)
    tris = _to_numpy(faces).astype(np.int64)
    if verts.ndim == 3:
        verts = verts[0]
    if tris.ndim == 3:
        tris = tris[0]
    if verts.ndim != 2 or verts.shape[-1] != 3:
        raise ValueError(f"vertices 应为 [V, 3]，实际为 {verts.shape}")
    if tris.ndim != 2 or tris.shape[-1] != 3:
        raise ValueError(f"faces 应为 [F, 3]，实际为 {tris.shape}")
    if tris.size and (tris.max() >= len(verts) or tris.min() < 0):
        raise ValueError("faces 中存在越界的顶点索引")
    return verts, tris


def _face_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """单位面法线 [F, 3]。"""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norm, 1e-8)


def _unique_edges_np(faces: np.ndarray) -> np.ndarray:
    """去重后的无向边 [E, 2]。"""
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _project(
    points: np.ndarray, azimuth: float, elevation: float
) -> Tuple[np.ndarray, np.ndarray]:
    """正交投影到图像平面（相机约定与 MultiViewRenderer 一致：y 轴向上）。

    Args:
        points: [P, 3] 世界坐标。
        azimuth: 方位角（度），0 度位于 +z 正面。
        elevation: 仰角（度）。

    Returns:
        (uv [P, 2] 图像坐标，右为 +u、下为 +v；depth [P] 距相机越远越大)
    """
    az, el = math.radians(azimuth), math.radians(elevation)
    eye = np.array(
        [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)],
        dtype=np.float32,
    )
    forward = -eye  # 相机看向原点
    right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    if np.linalg.norm(right) < 1e-6:  # 正对极点时换一个 up 参考
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    u = points @ right
    v = -(points @ up)  # 图像行方向朝下
    depth = -(points @ forward)
    return np.stack([u, v], axis=-1), depth


def _fit_to_canvas(
    uv: np.ndarray, image_size: int, margin: float = 0.08
) -> np.ndarray:
    """把投影坐标等比缩放平移到 [0, image_size) 画布内。"""
    low, high = uv.min(axis=0), uv.max(axis=0)
    span = float(np.max(high - low))
    if span <= 0:
        span = 1.0
    usable = image_size * (1.0 - 2.0 * margin)
    scaled = (uv - (low + high) * 0.5) * (usable / span)
    return scaled + image_size * 0.5


def visualize_normals(
    vertices: Any,
    faces: Any,
    output_path: str,
    azimuth: float = 35.0,
    elevation: float = 20.0,
    max_faces: int = 1500,
    image_size: int = 640,
    arrow_scale: float = 0.12,
) -> str:
    """可视化 mesh 法线方向。

    面法线以 RGB 编码（``(n + 1) / 2``）着色，并叠加一组法线箭头。

    Args:
        vertices: [V, 3] 或 [B, V, 3]。
        faces: [F, 3] 或 [B, F, 3]。
        output_path: 输出图片路径。
        azimuth/elevation: 观察角度（度）。
        max_faces: 最多绘制的法线箭头数量（等间隔抽样）。
        image_size: PIL 兜底路径下的画布边长。
        arrow_scale: 箭头长度（相对 mesh 尺度）。

    Returns:
        输出图片路径。
    """
    verts, tris = _mesh_arrays(vertices, faces)
    if tris.size == 0:
        raise ValueError("faces 为空，无法可视化法线")

    normals = _face_normals_np(verts, tris)
    centroids = verts[tris].mean(axis=1)
    step = max(1, len(tris) // max(1, int(max_faces)))
    sample = slice(None, None, step)
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    scale = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) * arrow_scale

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if _MATPLOTLIB_AVAILABLE:
        fig = plt.figure(figsize=(7, 7))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot_trisurf(
            verts[:, 0],
            verts[:, 2],
            verts[:, 1],
            triangles=tris,
            color=(0.75, 0.75, 0.78),
            alpha=0.35,
            linewidth=0.0,
            shade=True,
        )
        origins, directions = centroids[sample], normals[sample]
        axis.quiver(
            origins[:, 0],
            origins[:, 2],
            origins[:, 1],
            directions[:, 0] * scale,
            directions[:, 2] * scale,
            directions[:, 1] * scale,
            colors=colors[sample],
            linewidth=0.7,
        )
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(f"face normals ({len(tris)} faces)")
        axis.set_box_aspect((1, 1, 1))
        axis.set_xlabel("x")
        axis.set_ylabel("z")
        axis.set_zlabel("y")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    # ---- PIL 兜底：法线着色的点云 + 短箭头，按深度从远到近绘制 ----
    if not _PIL_AVAILABLE:
        raise RuntimeError("可视化法线需要 matplotlib 或 Pillow")

    uv_all, _ = _project(verts, azimuth, elevation)
    uv_c, depth = _project(centroids, azimuth, elevation)
    uv_tip, _ = _project(centroids + normals * scale, azimuth, elevation)
    # 质心与箭头端点必须与顶点共用同一套缩放平移，否则位置会错位
    low, high = uv_all.min(axis=0), uv_all.max(axis=0)
    span = float(np.max(high - low)) or 1.0
    factor = image_size * (1.0 - 0.16) / span
    center = (low + high) * 0.5

    def to_canvas(points: np.ndarray) -> np.ndarray:
        return (points - center) * factor + image_size * 0.5

    pts_c, pts_tip = to_canvas(uv_c), to_canvas(uv_tip)
    order = np.argsort(-depth)  # 远处先画，近处覆盖

    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index in order[::step]:
        color = tuple(int(c * 255) for c in colors[index])
        x0, y0 = pts_c[index]
        x1, y1 = pts_tip[index]
        draw.line([x0, y0, x1, y1], fill=color, width=1)
        draw.ellipse([x0 - 1, y0 - 1, x0 + 1, y0 + 1], fill=color)
    draw.text((8, 8), f"face normals ({len(tris)} faces)", fill=(0, 0, 0))
    return save_image(canvas, output_path)


def render_wireframe(
    vertices: Any,
    faces: Any,
    output_path: str,
    image_size: int = 640,
    azimuth: float = 35.0,
    elevation: float = 20.0,
    line_color: Tuple[int, int, int] = (40, 60, 110),
    background: Tuple[int, int, int] = (255, 255, 255),
    max_edges: int = 20000,
) -> str:
    """线框模式渲染 mesh（正交投影，不依赖 nvdiffrast / CUDA）。

    Args:
        vertices: [V, 3] 或 [B, V, 3]。
        faces: [F, 3] 或 [B, F, 3]。
        output_path: 输出图片路径。
        image_size: 画布边长。
        azimuth/elevation: 观察角度（度）。
        line_color: 线框颜色。
        background: 背景颜色。
        max_edges: 边数上限，超出时等间隔抽样以控制绘制耗时。

    Returns:
        输出图片路径。
    """
    verts, tris = _mesh_arrays(vertices, faces)
    if tris.size == 0:
        raise ValueError("faces 为空，无法渲染线框")

    edges = _unique_edges_np(tris)
    if len(edges) > max_edges:
        edges = edges[:: int(math.ceil(len(edges) / max_edges))]

    uv, _ = _project(verts, azimuth, elevation)
    points = _fit_to_canvas(uv, image_size)
    segments = points[edges]  # [E, 2, 2]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if _MATPLOTLIB_AVAILABLE:
        from matplotlib.collections import LineCollection

        fig, axis = plt.subplots(figsize=(image_size / 100.0, image_size / 100.0))
        axis.add_collection(
            LineCollection(
                segments,
                colors=[tuple(c / 255.0 for c in line_color)],
                linewidths=0.4,
            )
        )
        axis.set_xlim(0, image_size)
        axis.set_ylim(image_size, 0)  # 图像坐标：y 轴向下
        axis.set_aspect("equal")
        axis.set_facecolor(tuple(c / 255.0 for c in background))
        axis.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output_path, dpi=100, facecolor=axis.get_facecolor())
        plt.close(fig)
        return output_path

    if not _PIL_AVAILABLE:
        raise RuntimeError("线框渲染需要 matplotlib 或 Pillow")
    canvas = Image.new("RGB", (image_size, image_size), background)
    draw = ImageDraw.Draw(canvas)
    for (x0, y0), (x1, y1) in segments:
        draw.line([float(x0), float(y0), float(x1), float(y1)], fill=line_color, width=1)
    return save_image(canvas, output_path)


# ====================================================================== #
# 4. 论文对比图
# ====================================================================== #
def create_comparison_figure(
    images_dict: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None,
    ncols: Optional[int] = None,
) -> str:
    """创建对比 figure，用于论文展示。

    Args:
        images_dict: ``{"method_name": image_array, ...}``，图像可为张量 /
            numpy / PIL Image；多视图输入会先被拼成小网格。
        output_path: 输出图片路径。
        title: 图标题（可选）。
        ncols: 每行放几个方法，默认全部放一行。

    Returns:
        输出图片路径。
    """
    if not images_dict:
        raise ValueError("images_dict 为空，无法生成对比图")

    panels: Dict[str, np.ndarray] = {}
    for name, image in images_dict.items():
        if _PIL_AVAILABLE and isinstance(image, Image.Image):
            panels[name] = np.asarray(image.convert("RGB"))
            continue
        frames = to_uint8_images(image)
        panels[name] = frames[0] if frames.shape[0] == 1 else tile_images(frames)

    names = list(panels)
    ncols = max(1, int(ncols) if ncols else len(names))
    nrows = int(math.ceil(len(names) / ncols))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if _MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False
        )
        for index, axis in enumerate(axes.flatten()):
            axis.axis("off")
            if index < len(names):
                axis.imshow(panels[names[index]])
                axis.set_title(names[index], fontsize=11)
        if title:
            fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    # ---- PIL 兜底：统一缩放后横向 / 纵向平铺，并在每格上方写方法名 ----
    if not _PIL_AVAILABLE:
        raise RuntimeError("生成对比图需要 matplotlib 或 Pillow")
    cell = max(panel.shape[0] for panel in panels.values())
    label_h, padding = 20, 6
    canvas_w = ncols * cell + (ncols + 1) * padding
    canvas_h = nrows * (cell + label_h) + (nrows + 1) * padding + (24 if title else 0)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    offset_y = 24 if title else 0
    if title:
        draw.text((padding, 6), title, fill=(0, 0, 0))

    for index, name in enumerate(names):
        row, col = divmod(index, ncols)
        panel = Image.fromarray(panels[name]).resize((cell, cell), Image.BILINEAR)
        left = padding + col * (cell + padding)
        top = offset_y + padding + row * (cell + label_h + padding)
        draw.text((left, top), name, fill=(0, 0, 0))
        canvas.paste(panel, (left, top + label_h))
    return save_image(canvas, output_path)


if __name__ == "__main__":  # pragma: no cover - 手动自检入口
    import argparse

    parser = argparse.ArgumentParser(description="可视化工具自检 / 命令行入口")
    parser.add_argument("--log_file", type=str, default=None, help="绘制训练曲线的日志")
    parser.add_argument("--mesh", type=str, default=None, help="用于线框 / 法线图的 OBJ")
    parser.add_argument("--output_dir", type=str, default="./vis_out")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_file:
        path = plot_training_curves(
            args.log_file, os.path.join(args.output_dir, "curves.png"), smooth=5
        )
        print(f"训练曲线已保存: {path}")

    if args.mesh:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from datasets.shapenet import load_obj

        mesh_vertices, mesh_faces = load_obj(args.mesh)
        print(
            "线框图已保存: "
            + render_wireframe(
                mesh_vertices, mesh_faces, os.path.join(args.output_dir, "wireframe.png")
            )
        )
        print(
            "法线图已保存: "
            + visualize_normals(
                mesh_vertices, mesh_faces, os.path.join(args.output_dir, "normals.png")
            )
        )

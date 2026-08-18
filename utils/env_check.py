"""训练前的环境自检工具。

训练脚本依赖 CUDA、nvdiffrast（可微渲染）与 CLIP（语义先验）三条外部链路，
任意一条缺失都会在训练中途才暴露出来。本模块把这些检查前置到启动阶段，
并且一次性汇总全部失败项，附带可直接复制执行的修复命令。
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# 挂在 train.py 的 "semantic3d_gan" 之下，日志可直接写入 train.log
LOGGER = logging.getLogger("semantic3d_gan.env_check")

# 各项检查失败时给出的修复建议（用于最终的汇总报错）
FIX_HINTS: Dict[str, str] = {
    "cuda": (
        "安装带 CUDA 的 PyTorch，并确认 NVIDIA 驱动可用（nvidia-smi）：\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "  若只想跑 CPU 冒烟测试，请在配置中设置 training.device: cpu"
    ),
    "nvdiffrast": (
        "安装 nvdiffrast（需要 CUDA Toolkit 与 C++ 编译器）：\n"
        "    pip install ninja\n"
        "    pip install git+https://github.com/NVlabs/nvdiffrast.git"
    ),
    "clip": (
        "安装任一 CLIP 后端：\n"
        "    pip install open_clip_torch\n"
        "  或：pip install git+https://github.com/openai/CLIP.git"
    ),
    "data_root": (
        "检查配置中的 data.data_root 是否指向正确的数据集目录，"
        "目录下应能递归找到 .obj 或 .npy 网格文件"
    ),
}

# 递归查找网格文件时的最大扫描文件数，避免在超大数据集上卡住
_MAX_SCANNED_FILES = 20000

_MESH_SUFFIXES = (".obj", ".npy")


# ---------------------------------------------------------------------- #
# 状态输出
# ---------------------------------------------------------------------- #
def _status_marks() -> Tuple[str, str]:
    """返回 (成功标记, 失败标记)，终端编码不支持时退化为 ASCII。"""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✓✗".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return "[OK]", "[X]"
    return "[✓]", "[✗]"


_OK, _FAIL = _status_marks()


def _ensure_logging() -> None:
    """独立运行（未经 train.py 配置日志）时提供默认的控制台输出。"""
    if LOGGER.handlers or (LOGGER.parent is not None and LOGGER.parent.handlers):
        return
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    LOGGER.setLevel(logging.INFO)


def _ok(message: str) -> None:
    _ensure_logging()
    LOGGER.info("%s %s", _OK, message)


def _fail(message: str, hint: str = "") -> None:
    _ensure_logging()
    LOGGER.error("%s %s", _FAIL, message)
    if hint:
        for line in hint.splitlines():
            LOGGER.error("      %s", line)


def _info(message: str) -> None:
    _ensure_logging()
    LOGGER.info("      %s", message)


def _module_version(module: Any) -> str:
    """尽力获取模块版本号，取不到时返回 "unknown"。"""
    for attr in ("__version__", "version", "VERSION"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
    return "unknown"


# ---------------------------------------------------------------------- #
# 单项检查
# ---------------------------------------------------------------------- #
def check_cuda() -> bool:
    """验证 PyTorch 与 CUDA 可用性。

    Returns:
        CUDA 可用返回 True；torch 缺失或无可用 GPU 返回 False。
    """
    try:
        import torch
    except ImportError as exc:
        _fail(f"PyTorch 导入失败: {exc}", FIX_HINTS["cuda"])
        return False

    _info(f"torch {torch.__version__} (编译 CUDA: {torch.version.cuda or '无'})")

    if not torch.cuda.is_available():
        _fail("CUDA 不可用，可微渲染与训练无法在 GPU 上运行", FIX_HINTS["cuda"])
        return False

    device_count = torch.cuda.device_count()
    names = []
    for index in range(device_count):
        capability = torch.cuda.get_device_capability(index)
        names.append(
            f"{index}: {torch.cuda.get_device_name(index)} (sm_{capability[0]}{capability[1]})"
        )
    _ok(f"CUDA 可用，检测到 {device_count} 个 GPU")
    for name in names:
        _info(name)
    return True


def check_nvdiffrast() -> bool:
    """验证 nvdiffrast 可导入并报告版本。

    Returns:
        导入成功返回 True，否则 False。
    """
    try:
        nvdiffrast = importlib.import_module("nvdiffrast")
        importlib.import_module("nvdiffrast.torch")
    except ImportError as exc:
        _fail(f"nvdiffrast 导入失败: {exc}", FIX_HINTS["nvdiffrast"])
        return False

    _ok(f"nvdiffrast 可用（版本 {_module_version(nvdiffrast)}）")
    return True


def check_clip_backend() -> Optional[str]:
    """检测可用的 CLIP 后端。

    Returns:
        ``"official"``（OpenAI 官方 clip）、``"open_clip"``，或两者都缺失时返回
        ``None``。两者同时存在时返回 ``"official"``，与
        :mod:`vlm.clip_encoder` 的后端优先级保持一致。
    """
    available: List[str] = []

    try:
        official = importlib.import_module("clip")
        available.append("official")
        _info(f"OpenAI clip 可用（版本 {_module_version(official)}）")
    except ImportError:
        pass

    try:
        open_clip = importlib.import_module("open_clip")
        available.append("open_clip")
        _info(f"open_clip 可用（版本 {_module_version(open_clip)}）")
    except ImportError:
        pass

    if not available:
        _fail("未检测到任何 CLIP 后端，语义损失无法计算", FIX_HINTS["clip"])
        return None

    backend = "official" if "official" in available else "open_clip"
    _ok(f"CLIP 后端: {backend}")
    return backend


def check_data_root(path: str) -> bool:
    """验证数据目录存在且包含有效的网格文件（``.obj`` 或 ``.npy``）。

    Args:
        path: 数据集根目录，一般来自配置的 ``data.data_root``。

    Returns:
        目录存在且至少找到一个网格文件时返回 True。
    """
    if not path:
        _fail("配置中未指定 data.data_root", FIX_HINTS["data_root"])
        return False

    absolute = os.path.abspath(path)
    if not os.path.isdir(absolute):
        _fail(f"数据目录不存在: {absolute}", FIX_HINTS["data_root"])
        return False

    counts = {suffix: 0 for suffix in _MESH_SUFFIXES}
    scanned = 0
    truncated = False
    for _, _, files in os.walk(absolute):
        for name in files:
            scanned += 1
            lowered = name.lower()
            for suffix in _MESH_SUFFIXES:
                if lowered.endswith(suffix):
                    counts[suffix] += 1
        if scanned >= _MAX_SCANNED_FILES:
            truncated = True
            break

    total = sum(counts.values())
    if total == 0:
        _fail(
            f"数据目录 {absolute} 下未找到 .obj 或 .npy 文件（已扫描 {scanned} 个文件）",
            FIX_HINTS["data_root"],
        )
        return False

    summary = ", ".join(f"{suffix}: {counts[suffix]}" for suffix in _MESH_SUFFIXES)
    suffix_note = "+" if truncated else ""
    _ok(f"数据目录可用: {absolute}（{summary}{suffix_note}）")
    return True


# ---------------------------------------------------------------------- #
# 综合入口
# ---------------------------------------------------------------------- #
def run_all_checks(config: Dict[str, Any]) -> None:
    """训练前的综合环境检查。

    所有子项都会执行完毕，失败项一次性汇总后抛出，避免逐个报错反复重试。
    当配置中 ``training.device`` 为 CPU 时，CUDA 与 nvdiffrast 的失败降级为警告
    （CPU 只能用于冒烟测试，无法真正训练）。

    Args:
        config: ``train_config.yaml`` 解析出的配置字典。

    Raises:
        RuntimeError: 任一必需项检查失败，异常信息中附带具体修复步骤。
    """
    _ensure_logging()
    LOGGER.info("=" * 70)
    LOGGER.info("环境自检")
    LOGGER.info("=" * 70)
    _info(f"Python {sys.version.split()[0]} ({sys.executable})")

    config = config or {}
    device = str(config.get("training", {}).get("device", "cuda")).lower()
    gpu_required = not device.startswith("cpu")
    if not gpu_required:
        _info("training.device 为 CPU，GPU 相关检查仅作提示")

    failures: List[str] = []
    warnings: List[str] = []

    def record(name: str, passed: bool, required: bool = True) -> None:
        if passed:
            return
        (failures if required else warnings).append(name)

    record("cuda", check_cuda(), required=gpu_required)
    record("nvdiffrast", check_nvdiffrast(), required=gpu_required)
    record("clip", check_clip_backend() is not None)
    record("data_root", check_data_root(str(config.get("data", {}).get("data_root", ""))))

    if warnings:
        LOGGER.warning(
            "以下检查未通过，但当前为 CPU 模式，已跳过: %s", ", ".join(warnings)
        )

    if not failures:
        LOGGER.info("=" * 70)
        _ok("全部环境检查通过")
        LOGGER.info("=" * 70)
        return

    lines = [f"环境检查未通过（{len(failures)} 项）: {', '.join(failures)}"]
    for index, name in enumerate(failures, start=1):
        hint = FIX_HINTS.get(name, "请检查该项配置")
        lines.append(f"  {index}. [{name}] {hint}")
    lines.append("  也可一次性安装全部依赖: pip install -r requirements.txt")
    message = "\n".join(lines)
    LOGGER.error(message)
    raise RuntimeError(message)


def main() -> int:
    """命令行入口: ``python -m utils.env_check --config configs/train_config.yaml``。"""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic3D-GAN 环境自检")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "train_config.yaml"),
        help="训练配置文件路径",
    )
    args = parser.parse_args()

    config: Dict[str, Any] = {}
    try:
        import yaml
    except ImportError:
        _fail("PyYAML 未安装，无法读取配置: pip install pyyaml")
        return 1

    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    else:
        _fail(f"配置文件不存在: {args.config}")
        return 1

    try:
        run_all_checks(config)
    except RuntimeError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

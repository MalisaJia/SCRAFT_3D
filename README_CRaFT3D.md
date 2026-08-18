# CRaFT-3D（Critic-as-Reward Flow Tuning for 3D）

CRaFT-3D 是一个**纯 Diffusion 后端**的文本引导 3D 形状生成项目（已不再包含任何 GAN 训练路径）：以 **TripoSG**（VAST-AI，image-conditioned rectified-flow 扩散模型，输出 SDF 并经分层八叉树 marching cubes 提取 mesh）为骨干，通过 **LoRA 微调** DiT 注意力投影（`to_q` / `to_v`）适配目标数据分布；图像条件来自 **DINOv2**（pipeline 内置编码器），文本控制在推理阶段通过 **Learned Semantic Reward 种子重排序**实现 —— 采样多个扩散种子各自生成候选 mesh，渲染多视图后用 Learned Semantic Reward（继续教育后的 SemanticCritic，兼可选 CLIP 图文相似度）打分取最优。全流程统一 `{'vertices': [B, V, 3], 'faces': [F, 3]}` mesh 契约，渲染 / 评估 / 导出代码在训练、推理、打分三处零改动复用。

## 核心特性

- **`scripts/finetune_critic.py`**（Learned Semantic Reward 阶段一）：把已训练的 SemanticCritic 在真实 mesh 正例 + 损坏负例上继续教育，产出 `critic_v2.pt`。
- **`scripts/score_latents_critic.py`**（阶段二）：latent 缓存逐样本 decode -> 渲染 -> critic_v2 打分，产出训练权重 `critic_weights.json`（可断点续跑），供 `train_diffusion.py` 的 `training.critic_weights_path` 逐样本加权扩散 MSE。
- **Critic 重排（阶段三）**：`inference.py` 新增 `inference.scorer = clip / critic / combo` 与 `inference.critic_checkpoint`，扩散种子候选可用 critic_v2（及可选 CLIP 组合分）重排序取最优。
- **Reward3D hook**：`evaluate.py` 新增 `--reward3d_repo`，可懒加载外部 Reward3D 仓库为每个生成 mesh 追加独立分数列（缺依赖自动跳过，不影响其余指标）。
- **Objaverse 流式预计算**：`scripts/precompute_objaverse_stream.py` 边下载 glb 边编码 VAE latent + 渲染参考图 + DINOv2 图像条件，带磁盘守卫与断点续跑。
- **完整本地依赖图入包**：`evaluate.py` / `scripts/finetune_critic.py` / `scripts/score_latents_critic.py` 实际 import 的全部本地模块（`losses/`、`datasets/`、`utils/`、`vlm/`、`rendering/`、`models/`）均已包含，解压即可独立运行。
- **`run_reward_pipeline.sh`**：Learned Semantic Reward 三阶段串联管线参考脚本（服务器后台执行）。

## 文件结构

```text
CRaFT-3D/
├── README_CRaFT3D.md                  # 本文件
├── run_reward_pipeline.sh             # Learned Semantic Reward 三阶段串联管线（参考脚本）
├── train_diffusion.py                 # LoRA 微调主脚本：rectified flow 目标 + EMA + 梯度检查点 + 定期渲染评估
├── inference.py                       # diffusion 推理入口（文本 -> mesh + 多视图 + meta）
├── evaluate.py                        # 量化评估（CLIP Score / FID / 几何质量 / 可选 Reward3D）
├── models/
│   ├── __init__.py                    # 模型包入口；导出 DiffusionMeshGenerator 与 SemanticCritic
│   ├── diffusion_adapter.py           # TripoSG 适配器 DiffusionMeshGenerator：扩散采样 -> SDF -> 归一化 mesh 契约
│   └── semantic_critic.py             # SemanticCritic（Critic 重排 / 训练加权打分器）
├── losses/                            # 语义 / 几何损失包（evaluate.py 复用 geometry_reg；critic_loss 供 Critic 训练）
│   ├── __init__.py
│   ├── contrastive_loss.py
│   ├── critic_loss.py
│   ├── geometry_reg.py
│   ├── semantic_loss.py
│   └── view_diversity_loss.py
├── datasets/
│   ├── __init__.py
│   ├── objaverse.py                   # Objaverse 数据集 / 标注解析
│   └── shapenet.py                    # ShapeNet 数据集与 load_obj
├── rendering/
│   ├── __init__.py
│   └── multi_view_render.py           # nvdiffrast 多视角渲染器
├── utils/
│   ├── __init__.py
│   ├── anomaly_constructor.py         # 负样本 mesh 损坏构造（finetune_critic 负例）
│   ├── env_check.py                   # 环境自检（依赖 / 驱动 / 显存检查）
│   ├── geometry_features.py           # 几何统计描述子（Critic 输入）
│   └── visualize.py                   # 图像保存 / 拼图 / 曲线绘制
├── vlm/
│   ├── __init__.py
│   ├── clip_encoder.py                # 冻结 CLIP 编码器（打分 / 引导）
│   └── feature_fusion.py              # 视角感知 prompt 生成与多视角特征融合
├── scripts/
│   ├── precompute_latents.py          # 预计算 TripoSG VAE latent shard（含断点续跑与 DINOv2 图像嵌入）
│   ├── precompute_objaverse_stream.py # Objaverse 流式扩容预计算（边下载 glb 边编码 + 渲染 + DINOv2）
│   ├── finetune_critic.py             # Learned Semantic Reward 阶段一：Critic 继续教育 -> critic_v2.pt
│   ├── score_latents_critic.py        # Learned Semantic Reward 阶段二：latent 逐样本打分 -> 训练权重
│   ├── download_shapenet.sh           # ShapeNetCore 13 类流式下载（hf-mirror / HF token 双路径）
│   └── make_shapenet_annotations.py   # 生成与 Objaverse 同格式的 ShapeNet annotations.json
└── configs/
    ├── train_diffusion.yaml           # LoRA 训练配置（模型 / 训练 / 数据 / 日志四段）
    └── diffusion_inference.yaml       # diffusion 推理配置（采样参数 / 渲染 / CLIP / 种子重排序候选数 / 打分器）
```

各文件职责与关键符号：

| 文件 | 职责 | 关键类 / 函数 |
| --- | --- | --- |
| `models/__init__.py` | 模型包入口，直接导出扩散生成器与 Critic | `DiffusionMeshGenerator`、`SemanticCritic` |
| `models/diffusion_adapter.py` | 把 TripoSG pipeline 包装为与渲染器兼容的 mesh 生成器 | `DiffusionMeshGenerator`：`generate` / `set_text_to_image_hook` / `_load_pipeline` / `_load_lora_weights` / `_clip_seed_rerank` / `_sample_mesh` / `_postprocess` / `_canonicalize_mesh` / `_ensure_winding` / `_synthesize_condition_image` / `_octree_depth` |
| `train_diffusion.py` | TripoSG LoRA 微调训练循环（纯 latent，训练步内不渲染；可选 Learned Semantic Reward 逐样本损失加权） | `LoRALinear`、`LoRAEMA`、`LatentCacheDataset`、`inject_lora`、`lora_state_dict`、`load_lora_state_dict`、`build_dit_with_lora`、`forward_dit`、`train_step`、`sample_latent_euler`、`decode_latent_to_mesh`、`run_evaluation`、`save_checkpoint`、`load_checkpoint`、`main` |
| `scripts/precompute_latents.py` | 把 Objaverse/ShapeNet mesh 预编码为 VAE latent shard | `load_triposg_vae`、`_patch_vae_fps`、`_fps_fallback`、`load_dinov2_encoder`、`canonicalize_mesh`、`sample_surface_points`、`encode_mesh_latent`、`scan_existing_shards`、`flush_shard`、`find_reference_image`、`encode_image_embeds`、`main` |
| `scripts/finetune_critic.py` | Learned Semantic Reward 阶段一：旧 Critic 继续教育（真实 mesh 正例 + anomaly/新增损坏算子负例，冻结 CLIP 只训 MLP），产出与旧格式一致的 `critic_v2.pt` | `load_mesh_file`、`corrupt_vertex_noise`、`corrupt_face_flip`、`corrupt_decimate`、`apply_corruption`、`compute_sample_features`、`build_feature_pools`、`evaluate_pool`、`main` |
| `scripts/score_latents_critic.py` | Learned Semantic Reward 阶段二：latent 缓存逐样本 decode -> 渲染 -> critic_v2 打分，产出训练权重 `critic_weights.json`（可续跑） | `load_vae`、`scan_cache`、`scores_to_weights`、`main` |
| `scripts/download_shapenet.sh` | ShapeNetCore 13 类下载 / 解压 / 磁盘余量检查 / 汇总 | —（bash 脚本） |
| `scripts/make_shapenet_annotations.py` | 扫描 ShapeNet 目录生成 annotations JSON | `main` |
| `inference.py` | diffusion 推理：多种子搜索 + CLIP/Critic 重排 | `resolve_backend`、`build_diffusion_generator`、`build_renderer`、`build_clip`、`build_critic`、`search_seed`、`critic_score_for_mesh`、`clip_score_for_mesh`、`export_obj`、`generate`、`main` |
| `evaluate.py` | 量化评估（可选 Reward3D 外部分数列） | `mesh_collate`、`render_real_batch`、`generate_meshes_for_eval`、`clip_score_batch`、`InceptionFeatureExtractor`、`compute_fid`、`geometry_metrics`、`self_intersection_ratio`、`load_reward3d_scorer`、`reward3d_score_mesh`、`evaluate`、`main` |
| `configs/train_diffusion.yaml` | 训练配置 | `model.diffusion`（weights_path / lora_rank / lora_target_modules）、`training`（含 `critic_weights_path`）、`data`、`logging` |
| `configs/diffusion_inference.yaml` | 推理配置 | `model.diffusion`、`rendering`、`vlm`、`inference.num_candidates`、`inference.scorer`、`inference.critic_checkpoint`（建议 critic_v2） |

模块间调用关系：

- `inference.py` → `models/diffusion_adapter.py`（`build_diffusion_generator` 构建 `DiffusionMeshGenerator`，`search_seed` 调 `generate`）；→ `rendering/multi_view_render.py`、`vlm/clip_encoder.py`、`utils/visualize.py`（共享渲染 / CLIP / 图像工具）。
- `evaluate.py` → `inference.py`（复用 `build_diffusion_generator` / `search_seed` / `load_yaml_config` 等）；→ `datasets/*`、`losses/geometry_reg.py`（几何指标）；真实 mesh 的 collate 与渲染在本文件内实现（`mesh_collate` / `render_real_batch`）。
- `models/diffusion_adapter.py` → `train_diffusion.py`（`inject_lora` / `load_lora_state_dict` 加载微调权重）；→ `vlm/clip_encoder.py`（种子重排序打分）；→ `rendering/multi_view_render.py`（候选渲染）。
- `train_diffusion.py` → latent shard（`scripts/precompute_latents.py` 的输出）；评估阶段 → `rendering/multi_view_render.py`、`utils/visualize.py`；权重来自 `triposg`（TripoSG 仓库，懒加载）。
- `scripts/precompute_latents.py` → `datasets/objaverse.py` / `datasets/shapenet.py`（标注解析与 `load_obj`）；→ `triposg`（VAE / DINOv2）。
- `scripts/make_shapenet_annotations.py` → `datasets/shapenet.py`（`SHAPENET_SYNSET_IDS`）。

## 用法

### 1. 预计算 VAE latent（训练前一次性完成）

```bash
# 先准备 annotations（Objaverse 自带；ShapeNet 用下面脚本生成）
bash scripts/download_shapenet.sh                 # 可选：下载 ShapeNetCore 13 类
python scripts/make_shapenet_annotations.py \
    --data_root /path/to/shapenet --out /path/to/annotations.json

# mesh -> VAE latent shard（断点续跑，按 uid 跳过已处理样本）
python scripts/precompute_latents.py --config configs/train_diffusion.yaml \
    [--num_samples 2000] [--shard_size 256] [--device cuda]
# 输出：configs/train_diffusion.yaml 中 data.latent_cache_dir 下的
#       latent_shard_{i}.pt（latents/captions/uids[/image_embeds]）+ manifest.json
```

### 2. LoRA 微调训练

```bash
python train_diffusion.py --config configs/train_diffusion.yaml [--resume outputs/diffusion_train/ckpt_XXXXXXX.pt]
# 输出目录 outputs/diffusion_train/：
#   ckpt_*.pt          定期 checkpoint（LoRA + EMA + 优化器状态）
#   lora_weights.pt    最终纯 LoRA 权重（推理侧 model.diffusion.lora_weights_path 直接吃）
#   eval/sample_*.png  EMA 权重采样 -> mesh -> 多视图渲染预览
#   tb/、train.log     TensorBoard 与文本日志
```

### 3. 推理（文本 -> mesh）

```bash
# diffusion 后端（唯一后端）：全部参数来自 config + 命令行覆盖
python inference.py --config configs/diffusion_inference.yaml \
    --prompt "a red chair" --output_dir ./results \
    [--num_candidates 8] [--num_views 8] [--image_size 256] [--seed 0] [--device cuda]
# 输出 {prompt_slug}_mesh.obj / _views.png / _meta.json
```

### 4. 量化评估

```bash
python evaluate.py --config configs/diffusion_inference.yaml \
    --dataset objaverse --data_root ./data/objaverse \
    --num_samples 16 --num_candidates 4 --output report.json
# 可选：额外给每个生成 mesh 追加 Reward3D 外部分数列（缺依赖自动跳过）
python evaluate.py ... --reward3d_repo /path/to/Reward3D
```

## 语义奖励三阶段（Learned Semantic Reward / Critic-as-Reward）

CRaFT（Critic-as-Reward Flow Tuning）把已训练好的 SemanticCritic
升级为 **Learned Semantic Reward**，在扩散后端上分三个阶段落地：

```text
阶段一：继续教育               阶段二：训练加权                阶段三：推理重排
scripts/finetune_critic.py  -> scripts/score_latents_critic.py -> inference.py (scorer=critic/combo)
真实 mesh 正例                 latent cache 逐样本 decode         多种子候选生成
+ anomaly/新增损坏负例         -> 渲 4 视图 -> critic_v2 打分     -> 渲 4 视图 -> critic_v2 打分
冻结 CLIP，只训 Critic MLP     weight = clip(s, 0.5, 1.5)/mean    -> 取最优种子
产出 critic_v2.pt              产出 critic_weights.json           （任务 #31 已接入）
                                       |
                                       v
                          train_diffusion.py
                          training.critic_weights_path
                          逐样本加权扩散 MSE（缺失 uid 按 1.0）
```

1. **继续教育**（`scripts/finetune_critic.py`）：正例 = `--mesh_dir` 真实 mesh
   （可选 `--generate_triposg N` 用本地 TripoSG 补充）；负例 = 旧
   `AnomalyConstructor` 四策略 + 顶点高斯噪声 / 随机 decimate / 面片翻转
   新算子。渲染配置严格沿用旧 Critic checkpoint 内保存的训练配置
   （256 / 4 视图 / cam 2.5 / el ±30），冻结 CLIP、BCE 只训 Critic MLP，
   10% 源 mesh 留验证；输出 `critic_v2.pt`，格式与旧 checkpoint 完全一致
   （`{'critic', 'config'}`，`build_critic` 原样加载）。
2. **训练加权**（`scripts/score_latents_critic.py` + `train_diffusion.py`）：
   latent 缓存逐样本 VAE decode -> mesh -> critic_v2 打分，权重 = 分数软裁剪到
   [0.5, 1.5] 后整批均值归一到 1.0，写入 `cache_dir/critic_weights.json`
   （已打分 uid 自动跳过，可续跑，`--limit` 先跑子集）；训练时
   `training.critic_weights_path` 非空则按 uid 逐样本加权扩散 MSE，
   缺失 uid 按 1.0，日志记录覆盖率与 batch 权重统计。
3. **推理重排**（任务 #31 已接入）：`inference.scorer=critic/combo` 时用
   `critic_checkpoint`（建议指向 critic_v2）给多种子候选打分重排。

```bash
# 阶段一：继续教育（服务器上执行，需 GPU + nvdiffrast）
python scripts/finetune_critic.py --mesh_dir /path/to/meshes \
    --old_critic outputs/ablation_a0_full/ckpt_final.pt \
    --out outputs/critic_v2.pt --steps 3000
# 阶段二：latent 打分产出训练权重
python scripts/score_latents_critic.py \
    --cache_dir /root/autodl-tmp/3D-gans/cache/triposg_latents \
    --critic outputs/critic_v2.pt [--limit 256]
# 阶段二（训练）：把 training.critic_weights_path 指向上一步输出后照常训练
python train_diffusion.py --config configs/train_diffusion.yaml
# 阶段三：推理重排（把 inference.critic_checkpoint 改为 outputs/critic_v2.pt，
# inference.scorer 改为 critic/combo）
python inference.py --config configs/diffusion_inference.yaml --prompt "a red chair"
```

外部对照：`evaluate.py --reward3d_repo` 可懒加载 Reward3D 仓库给每个生成
mesh 追加独立分数列（依赖缺失时打印提示并跳过，不影响其余指标）。

## 依赖与环境要点

- **PyTorch ≥ 2.7（cu128）**：RTX 5090（sm_120 Blackwell）要求 cu128 wheel；旧 cu121 不支持该架构。
- **TripoSG 源码**：克隆 `VAST-AI-Research/TripoSG` 仓库并加入 `PYTHONPATH`（如 `site-packages/triposg_path.pth`）。本仓库对 TripoSG 全部**懒加载导入**，仅在真正构建生成器 / VAE 时才触发，因此 Critic 继续教育、几何指标等不涉及扩散采样的路径无需安装 TripoSG。
- **权重为 diffusers 目录布局**：`weights_path` 指向含 `model_index.json` 与 `vae/`、`transformer/`、`image_encoder_dinov2/`、`feature_extractor_dinov2/` 子目录的完整快照（HuggingFace repo 为 `VAST-AI/TripoSG`，约 7.5 GB）；留空则尝试在线拉取。
- **FPS 补丁**：上游 VAE 调用了被注释掉的 `torch_cluster.fps`，`precompute_latents.py` 用纯 PyTorch 贪心最远点采样给 `_sample_features` 打补丁（`_patch_vae_fps`），无需安装 torch_cluster。
- **sm_120 稳健性**：mesh 提取统一走 `use_flash_decoder=False` 的 naive 八叉树路径，不依赖需要即时编译的 CUDA flexicubes/flash 扩展。
- 其余依赖：`numpy`、`pyyaml`、`Pillow`；渲染需 `nvdiffrast`（CUDA）；CLIP 打分需 `open_clip/clip` 及权重；FID 需 `torchvision`；`tensorboard`、`tqdm` 可选。

## 纯 Diffusion 后端说明

- **单一生成路径**：项目已完全移除历史上的 GAN 训练与推理代码（`train.py`、`models/generator.py`、`models/discriminator.py`、`losses/adversarial_loss.py`、`utils/loss_scheduler.py`）。`train_diffusion.py` 是唯一训练入口，`inference.py` / `evaluate.py` 不再提供 `--backend` / `--checkpoint` 参数，也不再读取 `model.backend`。
- **checkpoint 语义**：diffusion 侧 checkpoint 只含 LoRA 权重（`{'lora', 'ema'}`）与可选优化器状态；`lora_weights.pt` 为最终纯 LoRA 权重，直接给推理配置的 `model.diffusion.lora_weights_path`。
- **Critic checkpoint 兼容**：`build_critic` 仍能加载历史训练产出的、含 `'critic'` 键的 checkpoint（包括 `scripts/finetune_critic.py` 继续教育后的 `critic_v2.pt`），格式一致、无需转换。
- **共享组件**：训练评估、推理重排、量化评估共用同一套 mesh 契约、`MultiViewRenderer`、`CLIPEncoder`、`export_obj` 与几何指标实现，评估口径一致。

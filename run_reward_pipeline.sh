#!/bin/bash
# 任务 #33 Learned Semantic Reward 串联管线 v2（服务器后台执行）：
#   打分器选型：critic_v2_rerun（val 0.6714/0.6857 > 旧版 0.5714，取更优者）
#   stage1 显存 >26GB 且预计算仍在跑时循环等待，直到预计算退出
#   stage2 score_latents_critic.py 打分 -> critic_weights.json
#   stage3 带权重条件重训 train_diffusion.py（3000 步）
# 每阶段时间戳写入 logs/reward_pipeline.log
set -u
cd /root/autodl-tmp/3D-gans
PY=/root/miniconda3/bin/python
PLOG=logs/reward_pipeline.log
CRITIC=outputs/critic_v2_rerun.pt
mkdir -p logs outputs

stamp() { echo "[$(date '+%F %T')] $*" >> "$PLOG"; }

stamp "===== reward pipeline v2 启动 (PID $$) | critic=$CRITIC ====="

# ---------- stage 1: 显存 / 预计算等待 ----------
USED=NA
while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$USED" ] || [ "$USED" -lt 26000 ] 2>/dev/null; then break; fi
    if ! pgrep -f 'precompute_objaverse_strea[m]' > /dev/null 2>&1; then break; fi
    stamp "stage1: 显存 ${USED}MiB > 26GB 且预计算在跑，120s 后重试"
    sleep 120
done
stamp "stage1: 显存检查通过（used=${USED} MiB）"

# ---------- stage 2: 打分 ----------
stamp "stage2: 打分开始 critic=$CRITIC"
$PY scripts/score_latents_critic.py \
    --cache_dir cache/triposg_latents_objaverse \
    --critic "$CRITIC" \
    --sharpen 3.0 \
    >> logs/score_latents.log 2>&1
SC=$?
CACHE=cache/triposg_latents_objaverse
N_SCORES=$($PY -c "import json;print(len(json.load(open('$CACHE/critic_scores.json'))))" 2>/dev/null || echo 0)
N_WEIGHTS=$($PY -c "import json;print(len(json.load(open('$CACHE/critic_weights.json'))))" 2>/dev/null || echo 0)
N_SAMPLES=$($PY -c "import json;print(len(json.load(open('$CACHE/manifest.json')).get('processed_uids',[])))" 2>/dev/null || echo 0)
stamp "stage2: 打分结束 exit=$SC | scores=$N_SCORES weights=$N_WEIGHTS / manifest 样本=$N_SAMPLES"

# 兜底：权重文件缺失 / 落后于 scores 时从 scores 重算（缺 uid 训练时按 1.0）
# 口径必须与主路径 score_latents_critic.py 的 scores_to_weights 完全一致：
# 归一 -> 1+3*(w-1) 锐化 -> clip[0.5,1.5] -> 乘性再归一（sharpen=3.0 同 stage2）
$PY - "$CACHE" <<'PYEOF' || stamp "stage2: 权重重算失败（不致命）"
import json, os, sys
cache = sys.argv[1]
sp = os.path.join(cache, "critic_scores.json")
wp = os.path.join(cache, "critic_weights.json")
if not os.path.isfile(sp):
    sys.exit(0)
scores = json.load(open(sp))
fresh = os.path.isfile(wp) and os.path.getmtime(wp) >= os.path.getmtime(sp)
if scores and not fresh:
    try:
        # 优先复用主路径函数，保证与 score_latents_critic.py 口径完全一致
        sys.path.insert(0, "scripts")
        from score_latents_critic import scores_to_weights
        weights = scores_to_weights(scores, sharpen=3.0)
    except Exception as exc:
        # 导入失败（如 torch/triposg 环境异常）时手工复刻 scores_to_weights
        # 四步：归一 -> 1+3*(w-1) -> clip[0.5,1.5] -> 乘性再归一。
        # 注意：此复刻必须与 score_latents_critic.py 保持同步！
        print("import scores_to_weights failed (%s), replicate inline" % exc)
        clipped = {u: min(max(float(v), 0.5), 1.5) for u, v in scores.items()}
        mean = sum(clipped.values()) / len(clipped)
        weights = ({u: v / mean for u, v in clipped.items()} if mean > 0
                   else {u: 1.0 for u in clipped})
        weights = {u: min(max(1.0 + 3.0 * (w - 1.0), 0.5), 1.5)
                   for u, w in weights.items()}
        sharp_mean = sum(weights.values()) / len(weights)
        if sharp_mean > 0:
            weights = {u: w / sharp_mean for u, w in weights.items()}
    with open(wp, "w") as fh:
        json.dump(weights, fh)
    print("weights regenerated:", len(weights))
PYEOF
N_WEIGHTS=$($PY -c "import json;print(len(json.load(open('$CACHE/critic_weights.json'))))" 2>/dev/null || echo 0)

# 门控：以续跑状态文件 scores 为准；打分进程崩了但已打分 >= 样本数 50% 也继续
HALF=$(( ${N_SAMPLES:-0} / 2 ))
if [ "${N_WEIGHTS:-0}" -ge 100 ] && [ "${N_SCORES:-0}" -ge "${HALF:-100}" ]; then
    stamp "stage2: 门控通过 weights=$N_WEIGHTS scores=$N_SCORES(门槛 $HALF)，继续训练；无权重 uid 按 1.0"
elif [ "${N_SCORES:-0}" -ge "${HALF:-100}" ]; then
    stamp "stage2: 门控放宽通过 scores=$N_SCORES(门槛 $HALF)，继续训练；无权重 uid 按 1.0"
else
    stamp "stage2: 打分结果过少（scores=$N_SCORES < $HALF），管线中止"
    exit 1
fi

# ---------- stage 3: 带权重重训 ----------
# 训练显存 ~27GB+：先等预计算进程退出再开训，防显存争抢 OOM（同 stage1 风格）
while pgrep -f 'precompute_objaverse_strea[m]' > /dev/null 2>&1; do
    stamp "stage3: 预计算仍在跑，等它退出再开训（120s 重试）"
    sleep 120
done
stamp "stage3: 预计算已退出，启动带权重条件重训（max_iterations=3000, log=logs/train_diffusion_reward.log）"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_diffusion.py --config configs/train_diffusion.yaml \
    > logs/train_diffusion_reward.log 2>&1
TR=$?
stamp "stage3: 重训结束 exit=$TR"
stamp "===== reward pipeline v2 全部完成 ====="
exit 0

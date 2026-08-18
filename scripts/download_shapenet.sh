#!/bin/bash
# download_shapenet.sh — 流式下载 ShapeNetCore 13 类（hf-mirror 直链 / HF token 两条路径）。
#
# 用法：
#   nohup bash scripts/download_shapenet.sh > logs/download_shapenet.log 2>&1 &
# 可选：export HF_TOKEN=hf_xxx （ShapeNet/ShapeNetCore 为 gated 仓库，需账号在
#       https://huggingface.co/datasets/ShapeNet/ShapeNetCore 申请访问获批后生成的 token；
#       无 token 时脚本回退到 hf-mirror 直链，截至 2026-08-12 该直链对 zip 返回 403）。
#
# 行为：按优先级逐类 wget -c -> unzip -> rm zip；每类前检查磁盘余量
# （zip 大小 + 5G 余量），单类失败不中断整体，结尾打印汇总。
set -u
DATA=/root/autodl-tmp/data/shapenet
LOG() { echo "[$(date '+%H:%M:%S')] $*"; }

# synset:zip大小(MB)，优先级从高到低（与 datasets/shapenet.py 的 13 类一致）
ITEMS=(
  "03001627:1966"   # chair
  "04379243:1670"   # table
  "04256520:1290"   # sofa
  "03636649:750"    # lamp
  "02958343:5686"   # car
  "02691156:3359"   # airplane
  "02828884:377"    # bench
  "02933112:433"    # cabinet
  "04401088:310"    # telephone
  "03691459:530"    # loudspeaker
  "03211117:290"    # display
  "04530566:1310"   # watercraft
  "04090263:930"    # rifle
)

mkdir -p "$DATA"
AUTH_HDR=()
if [ -n "${HF_TOKEN:-}" ]; then
  source /etc/network_turbo >/dev/null 2>&1 || true
  BASE="https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main"
  AUTH_HDR=(--header "Authorization: Bearer $HF_TOKEN")
else
  BASE="https://hf-mirror.com/datasets/ShapeNet/ShapeNetCore/resolve/main"
fi

OK=(); FAIL=()
for item in "${ITEMS[@]}"; do
  SYN="${item%%:*}"; MB="${item##*:}"
  ZIP="$SYN.zip"
  if [ -d "$DATA/$SYN" ] && [ -z "$(find "$DATA/$SYN" -name '*.obj' | head -1)" ]; then
    LOG "$SYN 目录存在但无 obj，重新下载"
  fi
  if [ -d "$DATA/$SYN" ] && [ -n "$(find "$DATA/$SYN" -name '*.obj' | head -1)" ]; then
    LOG "$SYN 已存在，跳过"
    OK+=("$SYN(cached)"); continue
  fi
  AVAIL_MB=$(df -PM "$DATA" | awk 'NR==2{print $4}')
  NEED_MB=$(( MB + 5120 ))
  if [ "$AVAIL_MB" -lt "$NEED_MB" ]; then
    LOG "$SYN 磁盘不足（可用 ${AVAIL_MB}MB < 需要 ${NEED_MB}MB），停止后续类别"
    FAIL+=("$SYN(disk)"); continue
  fi
  LOG "$SYN 开始下载（约 ${MB}MB）"
  if ! (cd "$DATA" && wget -q -c "${AUTH_HDR[@]}" "$BASE/$ZIP"); then
    LOG "$SYN 下载失败，跳过"
    FAIL+=("$SYN(download)"); rm -f "$DATA/$ZIP"; continue
  fi
  LOG "$SYN 解压中"
  if ! unzip -q -o "$DATA/$ZIP" -d "$DATA"; then
    LOG "$SYN 解压失败，跳过"
    FAIL+=("$SYN(unzip)"); rm -f "$DATA/$ZIP"; continue
  fi
  rm -f "$DATA/$ZIP"
  N=$(find "$DATA/$SYN" -name '*.obj' | wc -l)
  LOG "$SYN 完成，模型数 $N"
  OK+=("$SYN($N)")
done

LOG "===== 汇总 ====="
LOG "成功: ${OK[*]:-无}"
LOG "失败: ${FAIL[*]:-无}"
df -h "$DATA" | tail -1

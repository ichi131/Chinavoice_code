#!/bin/bash
# =============================================================================
# 一键脚本：data_mixft_v5 + FireRedASR 最优超参 训练 → 推理 → CER 评估
# ---------------------------------------------------------------------------
# 流程：
#   Step 1. 转换 data_mixft_v5/train.jsonl → VC 字段格式（首次运行时执行；已存在则跳过）
#   Step 2. 通过 sweep 框架用 sweep_mixft_v5_best.yaml 启动"单候选完整训练+评估"
#   Step 3. 从 sweep_summary.md 抽取 overall CER 打印，供与 11.53% baseline 对比
#
# 用法：
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash examples_train/asr_chinavoices/sweep/run_finetune_mixft_v5_best.sh
#
# 可选环境变量：
#   SWEEP_DIR              自定义顶层扫描目录（默认自动生成时间戳目录）
#   RESUME=1               启用断点续跑（转换文件已存在也会跳过）
#   FORCE_CONVERT=1        强制重新转换 mix_v5（覆盖 VC_data_mixft_v5/*）
#   EPOCHS=<n>             覆盖 YAML 里的 epochs（默认 4）
#
# 注意：本脚本仅调用现有 run_sweep_asr_chinavoices.sh + tools/convert_...py，
#      不覆盖 run_finetune_asr_chinavoices.sh / run_sweep_asr_chinavoices.sh 等既有入口。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# 路径 / 环境
# ---------------------------------------------------------------------------
MIX_SRC=/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_mixft_v5/train.jsonl
VC_OUT_DIR=/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_mixft_v5
VC_OUT_JSONL=$VC_OUT_DIR/data_train_vc.jsonl
CONVERT_PY=$SCRIPT_DIR/tools/convert_mixft_v5_to_vc_format.py
SWEEP_CONFIG=$SCRIPT_DIR/configs/sweep_mixft_v5_best.yaml

PYTHON_BIN=${PYTHON_BIN:-/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S_H20/bin/python3.10}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3)
fi

echo "[mixft_v5] PROJECT_ROOT   = $PROJECT_ROOT"
echo "[mixft_v5] MIX_SRC        = $MIX_SRC"
echo "[mixft_v5] VC_OUT_JSONL   = $VC_OUT_JSONL"
echo "[mixft_v5] SWEEP_CONFIG   = $SWEEP_CONFIG"
echo "[mixft_v5] PYTHON_BIN     = $PYTHON_BIN"
echo "[mixft_v5] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<未设>}"

# ---------------------------------------------------------------------------
# Step 1: 转换 mix_v5 → VC 格式（幂等）
# ---------------------------------------------------------------------------
if [[ ! -f "$MIX_SRC" ]]; then
  echo "[mixft_v5][ERROR] 源数据不存在：$MIX_SRC" >&2
  exit 1
fi

mkdir -p "$VC_OUT_DIR"

if [[ -f "$VC_OUT_JSONL" && "${FORCE_CONVERT:-0}" != "1" ]]; then
  n_in=$(wc -l < "$MIX_SRC")
  n_out=$(wc -l < "$VC_OUT_JSONL")
  echo "[mixft_v5][Step 1] 复用已存在的转换文件：$VC_OUT_JSONL (输入 $n_in 行, 输出 $n_out 行)"
  echo "[mixft_v5][Step 1] 若需强制重转换，请使用 FORCE_CONVERT=1"
else
  echo "[mixft_v5][Step 1] 开始转换 data_mixft_v5/train.jsonl → VC 格式 ..."
  "$PYTHON_BIN" "$CONVERT_PY" \
    --input  "$MIX_SRC" \
    --output "$VC_OUT_JSONL"
  echo "[mixft_v5][Step 1] 转换完成"
fi

# 校验字段：抽第 1 行看必需字段是否齐全
"$PYTHON_BIN" - <<PYCHECK
import json, sys
p = "$VC_OUT_JSONL"
with open(p) as f:
    line = f.readline()
d = json.loads(line)
need = ["key", "wav_path", "text", "accent"]
missing = [k for k in need if k not in d or not d[k]]
if missing:
    print(f"[mixft_v5][Step 1][ERROR] 转换后首行缺字段 {missing}: {d}", file=sys.stderr)
    sys.exit(2)
print(f"[mixft_v5][Step 1] 字段校验 OK. sample.key={d['key']} accent={d['accent']} text[:20]={d['text'][:20]}")
PYCHECK

# ---------------------------------------------------------------------------
# Step 2: 走 sweep 框架单候选跑训练 + 推理 + 评估
# ---------------------------------------------------------------------------
echo ""
echo "[mixft_v5][Step 2] 启动 sweep（单候选=最优超参组合）..."

# 允许通过 EPOCHS 环境变量覆盖 YAML 里的 epochs（临时改写一份运行时 yaml）
RUNTIME_YAML="$SWEEP_CONFIG"
if [[ -n "${EPOCHS:-}" ]]; then
  RUNTIME_YAML="$SCRIPT_DIR/configs/.sweep_mixft_v5_best.runtime.yaml"
  "$PYTHON_BIN" - <<PYWRITE
import re
src = "$SWEEP_CONFIG"
dst = "$RUNTIME_YAML"
epochs = int("$EPOCHS")
with open(src) as f:
    text = f.read()
text = re.sub(r"epochs:\s*\d+", f"epochs: {epochs}", text, count=1)
with open(dst, "w") as f:
    f.write(text)
print(f"[mixft_v5][Step 2] 已生成 runtime yaml：{dst}  (epochs={epochs})")
PYWRITE
fi

export SWEEP_CONFIG="$RUNTIME_YAML"
# SWEEP_DIR / RESUME 若外部设置就透传，未设则由 run_sweep_asr_chinavoices.sh 自动处理
if [[ -n "${SWEEP_DIR:-}" ]]; then export SWEEP_DIR; fi
if [[ -n "${RESUME:-}" ]];   then export RESUME;   fi

bash "$SCRIPT_DIR/run_sweep_asr_chinavoices.sh"

# ---------------------------------------------------------------------------
# Step 3: 结果汇总
# ---------------------------------------------------------------------------
echo ""
echo "[mixft_v5][Step 3] 抽取本次 sweep 结果 ..."

# 定位最新的 asr_mixft_v5_best_* 目录
LATEST_DIR=$(ls -td "$PROJECT_ROOT/exp/asr_mixft_v5_best_"* 2>/dev/null | head -1 || true)
if [[ -z "$LATEST_DIR" || ! -d "$LATEST_DIR" ]]; then
  echo "[mixft_v5][Step 3][WARN] 未找到 asr_mixft_v5_best_* 目录" >&2
  exit 0
fi

echo "[mixft_v5][Step 3] 最新 sweep 目录：$LATEST_DIR"

if [[ -f "$LATEST_DIR/sweep_summary.md" ]]; then
  echo "========== sweep_summary.md 尾部关键段 =========="
  tail -30 "$LATEST_DIR/sweep_summary.md"
  echo "================================================="
fi

# 打印本次 vs baseline（11.53%）
if [[ -f "$LATEST_DIR/sweep_summary.md" ]]; then
  new_cer=$(grep -Eo "overall CER = [0-9.]+" "$LATEST_DIR/sweep_summary.md" | tail -1 | awk '{print $NF}')
  if [[ -n "$new_cer" ]]; then
    echo ""
    echo "===== 结果对比 ====="
    echo "VC_data_v2 sweep 最优 (baseline) : 11.53 % overall CER"
    echo "data_mixft_v5 本次 (最优超参)    : $new_cer % overall CER"
    echo "===================="
  fi
fi

echo "[mixft_v5] Done."

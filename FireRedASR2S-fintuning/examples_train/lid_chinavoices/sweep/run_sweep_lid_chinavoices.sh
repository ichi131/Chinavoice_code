#!/bin/bash
# =============================================================================
# LID 超参贪心式扫描 · 入口 shell 脚本
# ---------------------------------------------------------------------------
# 用法：
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples_train/lid_chinavoices/sweep/run_sweep_lid_chinavoices.sh
#
# 可选环境变量：
#   SWEEP_CONFIG   YAML 配置路径（默认使用 configs/sweep_example.yaml）
#   SWEEP_DIR      顶层扫描目录（默认自动生成 exp/lid_sweep_<ts>/）
#   RESUME=1       启用断点恢复，跳过已完成的实验
#
# 隔离说明：本脚本仅位于 sweep/ 子目录下，不覆盖或修改任何现有 shell 脚本。
# =============================================================================
set -euo pipefail

# 项目根目录（相对本脚本 3 层上级：sweep/ -> lid_chinavoices/ -> examples_train/ -> <root>）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

# 与现有 run_finetune_lid_chinavoices.sh 保持一致的 conda 环境
export PATH=/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S_H20/bin:$PATH
export PYTHONPATH="$PROJECT_ROOT/fireredasr2s:${PYTHONPATH:-}"

# 默认使用 0-7 号 GPU；调用方可通过环境变量覆盖
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SWEEP_CONFIG=${SWEEP_CONFIG:-"$SCRIPT_DIR/configs/sweep_example.yaml"}
extra_args=()
if [[ -n "${SWEEP_DIR:-}" ]]; then
  extra_args+=("--sweep_dir" "$SWEEP_DIR")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  extra_args+=("--resume")
fi

echo "[run_sweep] PROJECT_ROOT=$PROJECT_ROOT"
echo "[run_sweep] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_sweep] SWEEP_CONFIG=$SWEEP_CONFIG"
echo "[run_sweep] extra_args=${extra_args[*]:-<none>}"

# 通过 python3.10 直接调用主控脚本（内部按 CUDA 卡数自动选择 python / torchrun 启动训练子进程）
exec python3.10 "$SCRIPT_DIR/sweep_main.py" \
  --config "$SWEEP_CONFIG" \
  "${extra_args[@]}"

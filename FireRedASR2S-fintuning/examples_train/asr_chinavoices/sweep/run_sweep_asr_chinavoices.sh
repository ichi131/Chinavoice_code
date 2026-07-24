#!/bin/bash
# =============================================================================
# ASR 超参贪心式扫描 · 入口 shell 脚本
# ---------------------------------------------------------------------------
# 用法：
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash examples_train/asr_chinavoices/sweep/run_sweep_asr_chinavoices.sh
#
# 可选环境变量：
#   SWEEP_CONFIG   YAML 配置路径（默认使用 configs/sweep_example.yaml）
#   SWEEP_DIR      顶层扫描目录（默认自动生成 exp/asr_sweep_<ts>/）
#   RESUME=1       启用断点恢复，跳过已完成的实验
#
# 隔离说明：本脚本仅位于 sweep/ 子目录下，不覆盖或修改任何现有 shell 脚本；
#          所有产物写入用户目录 exp/asr_sweep_*，绝不写到 user_ichiwang 目录。
# =============================================================================
set -euo pipefail

# 项目根目录（相对本脚本 3 层上级：sweep/ -> asr_chinavoices/ -> examples_train/ -> <root>）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

# 与现有 run_finetune_asr_chinavoices.sh 保持一致的 conda 环境
export PATH=/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S_H20/bin:$PATH
export PYTHONPATH="$PROJECT_ROOT/fireredasr2s:${PYTHONPATH:-}"

# 默认使用 0-7 号 GPU；调用方可通过环境变量覆盖
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# 一些常见的稳定性环境变量（分布式训练场景已在 LID 那边验证有效）
# 注：PyTorch 2.2+ 把 NCCL_ASYNC_ERROR_HANDLING 重命名为 TORCH_NCCL_ASYNC_ERROR_HANDLING，
# 两者语义等价；旧名会打 deprecated warning，这里直接用新名。
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

SWEEP_CONFIG=${SWEEP_CONFIG:-"$SCRIPT_DIR/configs/sweep_example.yaml"}
extra_args=()
if [[ -n "${SWEEP_DIR:-}" ]]; then
  extra_args+=("--sweep_dir" "$SWEEP_DIR")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  extra_args+=("--resume")
fi

echo "[run_sweep_asr] PROJECT_ROOT=$PROJECT_ROOT"
echo "[run_sweep_asr] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_sweep_asr] SWEEP_CONFIG=$SWEEP_CONFIG"
echo "[run_sweep_asr] SWEEP_DIR=${SWEEP_DIR:-<auto>}"
echo "[run_sweep_asr] RESUME=${RESUME:-0}"
echo "[run_sweep_asr] extra_args=${extra_args[*]:-<none>}"

# 通过 python3.10 -u 保证日志实时刷新；主控内部按 nproc_per_node 拉起 torchrun 训练子进程
exec python3.10 -u "$SCRIPT_DIR/sweep_main.py" \
  --config "$SWEEP_CONFIG" \
  "${extra_args[@]}"

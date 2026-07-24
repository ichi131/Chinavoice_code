#!/usr/bin/env bash
# =============================================================================
# infer_evalset_hpo_best.sh
# -----------------------------------------------------------------------------
# 用 VC_v2 HPO 搜出的最优 ckpt 对比赛 evaluation_set 做 8 卡推理并生成提交产物。
#
# 最优组合（来自 outputs_hpo/vc_v2/best.json）：
#   lr=3e-5, bs=8(global=256), warmup=0.03, sched=cosine,
#   esp=3, epochs=5, seed=42
#   → LID acc = 85.51%  (baseline: 83.44%,  +2.07pp)
#   → Overall CER = 15.23%
#
# 用法（默认参数已按最优 ckpt 写好）：
#   bash challenge_full_ft/infer_evalset_hpo_best.sh 2>&1 \
#       | tee challenge_full_ft/infer_logs/infer_evalset_hpo_best.log
#
# 覆盖示例（比如换 wav.scp 或输出目录）：
#   EVAL_SCP=/other/wav.scp OUT_DIR=/other/dir \
#     bash challenge_full_ft/infer_evalset_hpo_best.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# ---- HPO 最优 ckpt（硬编码为默认；可通过环境变量覆盖） ----
export MODEL_CKPT="${MODEL_CKPT:-/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/outputs_hpo/vc_v2/trial_lr3e-5_bs8_wu0.03_sched-cosine_esp3_ep5_seed42/checkpoint-350}"

# ---- 评估集 wav.scp（沿用 infer_evalset.sh 的默认值） ----
export EVAL_SCP="${EVAL_SCP:-/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/data/chinavoices_challenge/evaluation_set/wav.scp}"

# ---- 输出目录：独立目录，避免覆盖之前 baseline 的推理产物 ----
export OUT_DIR="${OUT_DIR:-/mnt/geminihzceph/user_johannapeng/challenge_model/infer_data_hpo_best}"

# ---- 推理超参（可按需覆盖） ----
export NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export CHECK_AUDIO="${CHECK_AUDIO:-0}"

echo "============================================================"
echo "[infer_evalset_hpo_best] 使用 HPO 最优 ckpt 推理"
echo "  MODEL_CKPT = ${MODEL_CKPT}"
echo "  EVAL_SCP   = ${EVAL_SCP}"
echo "  OUT_DIR    = ${OUT_DIR}"
echo "  NUM_GPUS   = ${NUM_GPUS}"
echo "  BATCH_SIZE = ${BATCH_SIZE}"
echo "============================================================"

# 前置检查：确认最优 ckpt 存在
if [[ ! -d "${MODEL_CKPT}" ]]; then
    echo "[infer_evalset_hpo_best][ERROR] MODEL_CKPT 目录不存在: ${MODEL_CKPT}" >&2
    exit 1
fi

# 复用现有 3 步管线
exec bash "${SCRIPT_DIR}/infer_evalset.sh"

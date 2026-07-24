#!/usr/bin/env bash
# =============================================================================
# train_full_ft.sh
# -----------------------------------------------------------------------------
# Qwen3-ASR 全参数微调（Full-Parameter FT，不使用 LoRA）训练入口 shell。
#
# 所有关键参数均以 shell 变量暴露，可通过 `A=xx B=yy bash train_full_ft.sh`
# 或先 `export` 后再运行来覆盖。默认路径指向 `challenge_full_ft/data` 与
# `challenge_full_ft/outputs`。
#
# ⚠️ 只应在带 GPU 的实验机上运行，不要在无 GPU 环境执行。
# =============================================================================

set -euo pipefail

# ---- 目录 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---- 路径 ----
# ⚠️ MODEL_PATH 建议使用**本地绝对路径**（含 preprocessor_config.json 等的完整 base
# 目录），这样 MakeEveryCheckpointInferableCallback 才能正确把 processor 文件
# 拷贝到每个 checkpoint-*。若传 HF Hub ID（如 Qwen/Qwen3-ASR-1.7B），python
# 侧会自动用 snapshot_download 解析成本地路径，但需要联网/HF cache 存在。
MODEL_PATH=${MODEL_PATH:-"/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B"}
TRAIN_FILE=${TRAIN_FILE:-"${SCRIPT_DIR}/data/train.jsonl"}
EVAL_FILE=${EVAL_FILE:-"${SCRIPT_DIR}/data/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs"}

# ---- 训练超参 ----
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER=${LR_SCHEDULER:-"cosine"}
LOG_STEPS=${LOG_STEPS:-10}
SR=${SR:-16000}

# ---- 保存 / eval ----
# 说明：当前训练集约 3.4w 条，8 卡 × BATCH_SIZE=8 × GRAD_ACC=4 时
#   - 全局 batch = 256，1 epoch ≈ 134 步，5 epoch ≈ 670 步
#   - SAVE_STEPS=50 → 5 epoch 内触发 ~13 次 eval，早停/挑最优才有意义
#   - SAVE_TOTAL_LIMIT=2 → 只保留"当前最佳 + 最近一次"，磁盘占用可控
# 若后期换数据/换 batch 规模，请同步在命令行覆盖这两个变量。
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_STEPS=${EVAL_STEPS:-0}          # 0 表示与 SAVE_STEPS 对齐
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}

# ---- 精度 ----
BF16=${BF16:-1}

# ---- 早停 ----
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
EARLY_STOP_THRESHOLD=${EARLY_STOP_THRESHOLD:-0.0}

# ---- 随机种子（HPO 搜索时用） ----
SEED=${SEED:-42}

# ---- 分布式 ----
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}

# ---- DataLoader ----
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}

# ---- Resume ----
RESUME=${RESUME:-0}
RESUME_FROM=${RESUME_FROM:-""}

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "[train_full_ft] MODEL_PATH            = ${MODEL_PATH}"
echo "[train_full_ft] TRAIN_FILE            = ${TRAIN_FILE}"
echo "[train_full_ft] EVAL_FILE             = ${EVAL_FILE}"
echo "[train_full_ft] OUTPUT_DIR            = ${OUTPUT_DIR}"
echo "[train_full_ft] EPOCHS                = ${EPOCHS}"
echo "[train_full_ft] BATCH_SIZE            = ${BATCH_SIZE}"
echo "[train_full_ft] GRAD_ACC              = ${GRAD_ACC}"
echo "[train_full_ft] LR                    = ${LR}"
echo "[train_full_ft] WARMUP_RATIO          = ${WARMUP_RATIO}"
echo "[train_full_ft] LR_SCHEDULER          = ${LR_SCHEDULER}"
echo "[train_full_ft] SAVE_STEPS            = ${SAVE_STEPS}"
echo "[train_full_ft] EVAL_STEPS            = ${EVAL_STEPS}"
echo "[train_full_ft] SAVE_TOTAL_LIMIT      = ${SAVE_TOTAL_LIMIT}"
echo "[train_full_ft] BF16                  = ${BF16}"
echo "[train_full_ft] EARLY_STOP_PATIENCE   = ${EARLY_STOP_PATIENCE}"
echo "[train_full_ft] EARLY_STOP_THRESHOLD  = ${EARLY_STOP_THRESHOLD}"
echo "[train_full_ft] SEED                  = ${SEED}"
echo "[train_full_ft] NUM_GPUS              = ${NUM_GPUS}"
echo "============================================================"

CMD_ARGS=(
    --model_path            "${MODEL_PATH}"
    --train_file            "${TRAIN_FILE}"
    --eval_file             "${EVAL_FILE}"
    --output_dir            "${OUTPUT_DIR}"
    --sr                    "${SR}"
    --batch_size            "${BATCH_SIZE}"
    --grad_acc              "${GRAD_ACC}"
    --lr                    "${LR}"
    --epochs                "${EPOCHS}"
    --log_steps             "${LOG_STEPS}"
    --lr_scheduler_type     "${LR_SCHEDULER}"
    --warmup_ratio          "${WARMUP_RATIO}"
    --num_workers           "${NUM_WORKERS}"
    --pin_memory            "${PIN_MEMORY}"
    --persistent_workers    "${PERSISTENT_WORKERS}"
    --prefetch_factor       "${PREFETCH_FACTOR}"
    --save_steps            "${SAVE_STEPS}"
    --eval_steps            "${EVAL_STEPS}"
    --save_total_limit      "${SAVE_TOTAL_LIMIT}"
    --bf16                  "${BF16}"
    --early_stopping_patience  "${EARLY_STOP_PATIENCE}"
    --early_stopping_threshold "${EARLY_STOP_THRESHOLD}"
    --seed                  "${SEED}"
    --resume                "${RESUME}"
)
if [[ -n "${RESUME_FROM}" ]]; then
    CMD_ARGS+=(--resume_from "${RESUME_FROM}")
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    LAUNCHER=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}")
else
    LAUNCHER=(python)
fi

set -x
"${LAUNCHER[@]}" "${SCRIPT_DIR}/qwen3_asr_sft_full.py" "${CMD_ARGS[@]}"
set +x

echo "[train_full_ft] done. best_ckpt.txt (if any) -> ${OUTPUT_DIR}/best_ckpt.txt"

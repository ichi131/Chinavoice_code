#!/usr/bin/env bash
# =============================================================================
# run_mixft_v6_pipeline.sh
# -----------------------------------------------------------------------------
# 一键跑通「混合方言联合微调 v6」的全流程
# 相对 v5 的唯一改动：对困难方言（wuyu/kejia/chaoshan/nanchang/minnan）
# 的 VC_v2 子集做 2× 重采样；训练超参、评估流程完全继承 v5。
#
#   Step 1  prepare_mixft_v6_data.py：v5 数据 + 困难方言 VC_v2 子集 2×
#           输出到 data_mixft_v6/
#   Step 2  train_full_ft.sh：8 卡训练，EPOCHS=3、SAVE_STEPS=200
#           输出到 outputs_mixft_v6/
#   Step 3  run_eval.sh：挑最佳 ckpt -> 8 卡推理官方 test -> 计算 CER
#   Step 4  打印最终 CER / 按方言 CER / LID 准确率
#
# 使用：
#   bash challenge_full_ft/run_mixft_v6_pipeline.sh
# 或（把日志同时落到文件）：
#   bash challenge_full_ft/run_mixft_v6_pipeline.sh 2>&1 | tee run_mixft_v6_pipeline.log
#
# 环境变量：
#   SKIP_PREPARE=1     跳过 Step 1，直接复用已有 data_mixft_v6/
#   HARD_ACCENTS=...   逗号分隔的困难方言列表（默认 wuyu,kejia,chaoshan,nanchang,minnan）
#   HARD_DUP=1         困难方言额外复制次数（=1 → 2×，=2 → 3×，默认 1）
# =============================================================================
set -euo pipefail

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SFT_DIR=${SFT_DIR:-"${SCRIPT_DIR}/data_mixft_v6"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs_mixft_v6"}

MODEL_PATH=${MODEL_PATH:-"/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B"}

OFFICIAL_TEST=${OFFICIAL_TEST:-"${SCRIPT_DIR}/data/test.jsonl"}

# ---------- 数据准备参数 ----------
PER_ACCENT_MAX=${PER_ACCENT_MAX:-25000}
SEED=${SEED:-42}
SKIP_PREPARE=${SKIP_PREPARE:-0}

# v6 新增：困难样本 2× 重采样配置
HARD_ACCENTS=${HARD_ACCENTS:-"wuyu,kejia,chaoshan,nanchang,minnan"}
HARD_DUP=${HARD_DUP:-1}

# ---------- 训练超参（完全继承 v5） ----------
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER=${LR_SCHEDULER:-"cosine"}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
EARLY_STOP_THRESHOLD=${EARLY_STOP_THRESHOLD:-0.0}

# ---------- 打印配置 ----------
echo "============================================================"
echo "[run_mixft_v6] SFT_DIR         = ${SFT_DIR}"
echo "[run_mixft_v6] OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "[run_mixft_v6] MODEL_PATH      = ${MODEL_PATH}"
echo "[run_mixft_v6] OFFICIAL_TEST   = ${OFFICIAL_TEST}"
echo "[run_mixft_v6] PER_ACCENT_MAX  = ${PER_ACCENT_MAX}"
echo "[run_mixft_v6] SEED            = ${SEED}"
echo "[run_mixft_v6] SKIP_PREPARE    = ${SKIP_PREPARE}"
echo "[run_mixft_v6] HARD_ACCENTS    = ${HARD_ACCENTS}"
echo "[run_mixft_v6] HARD_DUP        = ${HARD_DUP}  (occurrences = $((1 + HARD_DUP))x)"
echo "------------------------------------------------------------"
echo "[run_mixft_v6] NUM_GPUS        = ${NUM_GPUS}"
echo "[run_mixft_v6] EPOCHS          = ${EPOCHS}"
echo "[run_mixft_v6] BATCH_SIZE      = ${BATCH_SIZE}"
echo "[run_mixft_v6] GRAD_ACC        = ${GRAD_ACC}"
echo "[run_mixft_v6] LR              = ${LR}"
echo "[run_mixft_v6] WARMUP_RATIO    = ${WARMUP_RATIO}"
echo "[run_mixft_v6] LR_SCHEDULER    = ${LR_SCHEDULER}"
echo "[run_mixft_v6] SAVE_STEPS      = ${SAVE_STEPS}"
echo "[run_mixft_v6] SAVE_TOTAL_LIMIT      = ${SAVE_TOTAL_LIMIT}"
echo "[run_mixft_v6] EARLY_STOP_PATIENCE   = ${EARLY_STOP_PATIENCE}"
echo "[run_mixft_v6] EARLY_STOP_THRESHOLD  = ${EARLY_STOP_THRESHOLD}"
echo "============================================================"

# ---------- 前置检查 ----------
if [[ ! -f "${OFFICIAL_TEST}" ]]; then
    echo "[run_mixft_v6] ERROR: 官方 test 集不存在: ${OFFICIAL_TEST}" >&2
    exit 1
fi

# ---------- Step 1: 数据准备 ----------
echo
echo "[run_mixft_v6][1/3] prepare_mixft_v6_data.py -> ${SFT_DIR}"
if [[ "${SKIP_PREPARE}" == "1" && -s "${SFT_DIR}/train.jsonl" && -s "${SFT_DIR}/val.jsonl" ]]; then
    echo "[run_mixft_v6]     SKIP_PREPARE=1，跳过 prepare_mixft_v6_data.py"
    echo "[run_mixft_v6]     复用 ${SFT_DIR}"
else
    python "${SCRIPT_DIR}/prepare_mixft_v6_data.py" \
        --out_dir        "${SFT_DIR}" \
        --per_accent_max "${PER_ACCENT_MAX}" \
        --hard_accents   "${HARD_ACCENTS}" \
        --hard_dup       "${HARD_DUP}" \
        --seed           "${SEED}"
fi

SFT_TRAIN="${SFT_DIR}/train.jsonl"
SFT_VAL="${SFT_DIR}/val.jsonl"
if [[ ! -s "${SFT_TRAIN}" || ! -s "${SFT_VAL}" ]]; then
    echo "[run_mixft_v6] ERROR: SFT jsonl 生成失败，train/val 为空。" >&2
    exit 1
fi
echo "[run_mixft_v6]     SFT train lines = $(wc -l < "${SFT_TRAIN}")"
echo "[run_mixft_v6]     SFT val   lines = $(wc -l < "${SFT_VAL}")"

# ---------- Step 2: 训练 ----------
echo
echo "[run_mixft_v6][2/3] 启动训练：${NUM_GPUS} 卡 × EPOCHS=${EPOCHS}"
mkdir -p "${OUTPUT_DIR}"
NUM_GPUS="${NUM_GPUS}" \
MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${SFT_TRAIN}" \
EVAL_FILE="${SFT_VAL}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EPOCHS="${EPOCHS}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACC="${GRAD_ACC}" \
LR="${LR}" \
WARMUP_RATIO="${WARMUP_RATIO}" \
LR_SCHEDULER="${LR_SCHEDULER}" \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE}" \
EARLY_STOP_THRESHOLD="${EARLY_STOP_THRESHOLD}" \
bash "${SCRIPT_DIR}/train_full_ft.sh"

# ---------- Step 3: 评估 ----------
echo
echo "[run_mixft_v6][3/3] 评估：挑最佳 ckpt -> 推理 ${OFFICIAL_TEST} -> 计算 CER"
PRED_JSONL="${OUTPUT_DIR}/pred_test.jsonl"
WER_DIR="${OUTPUT_DIR}/wer_eval"

OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_TEST="${OFFICIAL_TEST}" \
PRED_JSONL="${PRED_JSONL}" \
WER_DIR="${WER_DIR}" \
bash "${SCRIPT_DIR}/run_eval.sh"

# ---------- Step 4: 打印最终结果 ----------
echo
echo "============================================================"
echo "[run_mixft_v6] ★ 全流程完成，关键产物："
echo "  - 训练输出目录 : ${OUTPUT_DIR}"
echo "  - 最佳 ckpt    : $(cat "${OUTPUT_DIR}/best_ckpt.txt" 2>/dev/null || echo 'N/A')"
echo "  - 推理结果     : ${PRED_JSONL}"
echo "  - WER 目录     : ${WER_DIR}"
echo "------------------------------------------------------------"
if [[ -f "${WER_DIR}/result.wer" ]]; then
    echo "[整体 CER] (${WER_DIR}/result.wer):"
    cat "${WER_DIR}/result.wer"
fi
echo "------------------------------------------------------------"
if [[ -f "${WER_DIR}/by_dialect_summary.txt" ]]; then
    echo "[按方言 CER] (${WER_DIR}/by_dialect_summary.txt):"
    cat "${WER_DIR}/by_dialect_summary.txt"
fi
echo "------------------------------------------------------------"
if [[ -f "${WER_DIR}/dialect_accuracy.txt" ]]; then
    echo "[方言分类准确率] (${WER_DIR}/dialect_accuracy.txt):"
    cat "${WER_DIR}/dialect_accuracy.txt"
fi
echo "============================================================"

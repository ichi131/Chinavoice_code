#!/usr/bin/env bash
# =============================================================================
# run_mixft_v5_pipeline.sh
# -----------------------------------------------------------------------------
# 一键跑通「混合方言联合微调 v5」的全流程（去方言子标签版）：
#   Step 1  prepare_mixft_v5_data.py：加载 12 外部方言 + VC_v2 挑战集，
#           SFT target 重写为 `language Chinese<asr_text>xxx`（官方推荐格式），
#           按 accent 截断到 25000 条 → data_mixft_v5/
#   Step 2  train_full_ft.sh：8 卡训练，EPOCHS=3、SAVE_STEPS=200，
#           输出到 outputs_mixft_v5/
#   Step 3  run_eval.sh：挑最佳 ckpt → 8 卡推理官方 test → 计算 CER
#   Step 4  打印最终 CER / 按方言 CER / LID 准确率
#
# 使用：
#   bash challenge_full_ft/run_mixft_v5_pipeline.sh
# 或（把日志同时落到文件）：
#   bash challenge_full_ft/run_mixft_v5_pipeline.sh 2>&1 | tee run_mixft_v5_pipeline.log
#
# 支持环境变量覆盖，详见下方“路径 / 超参”。
#
# ❗与 v3 差异：
#   * SFT_DIR   默认 data_mixft_v5/
#   * OUTPUT_DIR 默认 outputs_mixft_v5/
#   * prepare 脚本换为 prepare_mixft_v5_data.py
#   * 推理侧不需任何改动：模型不再吐 "language Chinese xxx<asr_text>"。
#     infer_test.py 中 split_asr_content 的 else 分支已天然兼容 "无 tag也没关系"，
#     pred_dialect 会为空串，dialect_accuracy 会全 0（预期内）。
# =============================================================================
set -euo pipefail

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SFT_DIR=${SFT_DIR:-"${SCRIPT_DIR}/data_mixft_v5"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs_mixft_v5"}

MODEL_PATH=${MODEL_PATH:-"/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B"}

# 官方 test 集（与其他实验对齐）
OFFICIAL_TEST=${OFFICIAL_TEST:-"${SCRIPT_DIR}/data/test.jsonl"}

# ---------- 数据准备参数 ----------
PER_ACCENT_MAX=${PER_ACCENT_MAX:-25000}
SEED=${SEED:-42}
SKIP_PREPARE=${SKIP_PREPARE:-0}   # =1 时跳过 Step 1，直接复用已有 data_mixft_v5/

# ---------- 训练超参（在 VC_v2 基础上小改：EPOCHS/SAVE_STEPS） ----------
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
echo "[run_mixft_v5] SFT_DIR         = ${SFT_DIR}"
echo "[run_mixft_v5] OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "[run_mixft_v5] MODEL_PATH      = ${MODEL_PATH}"
echo "[run_mixft_v5] OFFICIAL_TEST   = ${OFFICIAL_TEST}"
echo "[run_mixft_v5] PER_ACCENT_MAX  = ${PER_ACCENT_MAX}"
echo "[run_mixft_v5] SEED            = ${SEED}"
echo "[run_mixft_v5] SKIP_PREPARE    = ${SKIP_PREPARE}"
echo "------------------------------------------------------------"
echo "[run_mixft_v5] NUM_GPUS        = ${NUM_GPUS}"
echo "[run_mixft_v5] EPOCHS          = ${EPOCHS}"
echo "[run_mixft_v5] BATCH_SIZE      = ${BATCH_SIZE}"
echo "[run_mixft_v5] GRAD_ACC        = ${GRAD_ACC}"
echo "[run_mixft_v5] LR              = ${LR}"
echo "[run_mixft_v5] WARMUP_RATIO    = ${WARMUP_RATIO}"
echo "[run_mixft_v5] LR_SCHEDULER    = ${LR_SCHEDULER}"
echo "[run_mixft_v5] SAVE_STEPS      = ${SAVE_STEPS}"
echo "[run_mixft_v5] SAVE_TOTAL_LIMIT      = ${SAVE_TOTAL_LIMIT}"
echo "[run_mixft_v5] EARLY_STOP_PATIENCE   = ${EARLY_STOP_PATIENCE}"
echo "[run_mixft_v5] EARLY_STOP_THRESHOLD  = ${EARLY_STOP_THRESHOLD}"
echo "============================================================"

# ---------- 前置检查 ----------
if [[ ! -f "${OFFICIAL_TEST}" ]]; then
    echo "[run_mixft_v5] ERROR: 官方 test 集不存在: ${OFFICIAL_TEST}" >&2
    exit 1
fi

# ---------- Step 1: 数据准备 ----------
echo
echo "[run_mixft_v5][1/3] prepare_mixft_v5_data.py -> ${SFT_DIR}"
if [[ "${SKIP_PREPARE}" == "1" && -s "${SFT_DIR}/train.jsonl" && -s "${SFT_DIR}/val.jsonl" ]]; then
    echo "[run_mixft_v5]     SKIP_PREPARE=1，跳过 prepare_mixft_v5_data.py"
    echo "[run_mixft_v5]     复用 ${SFT_DIR}"
else
    python "${SCRIPT_DIR}/prepare_mixft_v5_data.py" \
        --out_dir        "${SFT_DIR}" \
        --per_accent_max "${PER_ACCENT_MAX}" \
        --seed           "${SEED}"
fi

SFT_TRAIN="${SFT_DIR}/train.jsonl"
SFT_VAL="${SFT_DIR}/val.jsonl"
if [[ ! -s "${SFT_TRAIN}" || ! -s "${SFT_VAL}" ]]; then
    echo "[run_mixft_v5] ERROR: SFT jsonl 生成失败，train/val 为空。" >&2
    exit 1
fi
echo "[run_mixft_v5]     SFT train lines = $(wc -l < "${SFT_TRAIN}")"
echo "[run_mixft_v5]     SFT val   lines = $(wc -l < "${SFT_VAL}")"

# ---------- Step 2: 训练 ----------
echo
echo "[run_mixft_v5][2/3] 启动训练：${NUM_GPUS} 卡 × EPOCHS=${EPOCHS}"
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
echo "[run_mixft_v5][3/3] 评估：挑最佳 ckpt -> 推理 ${OFFICIAL_TEST} -> 计算 CER"
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
echo "[run_mixft_v5] ★ 全流程完成，关键产物："
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

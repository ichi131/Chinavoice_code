#!/usr/bin/env bash
# =============================================================================
# run_vc_pipeline.sh
# -----------------------------------------------------------------------------
# 一键跑通「VC 增广数据」的全流程：
#   Step 1  在 VC_data 下创建空 test 占位文件（prepare_data 强制要求三份都在）
#   Step 2  prepare_data.py：把原始 jsonl 转为 SFT 格式，输出到 data_vc/
#   Step 3  train_full_ft.sh：8 卡训练，超参严格对齐 baseline，输出到 outputs_vc/
#   Step 4  run_eval.sh：挑最佳 ckpt → 8 卡推理官方 test → 计算 CER
#   Step 5  打印最终 CER / 按方言 CER 结果
#
# 使用：
#   bash challenge_full_ft/run_vc_pipeline.sh
# 或（把日志同时落到文件）：
#   bash challenge_full_ft/run_vc_pipeline.sh 2>&1 | tee run_vc_pipeline.log
#
# 所有关键变量都可以通过环境变量覆盖，参见下方 "路径 / 超参" 部分。
# =============================================================================

set -euo pipefail

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VC_DATA_DIR=${VC_DATA_DIR:-"${ROOT_DIR}/VC_data"}
TRAIN_NAME=${TRAIN_NAME:-"data_train_vc.jsonl"}
VAL_NAME=${VAL_NAME:-"data_val_vc.jsonl"}
TEST_NAME=${TEST_NAME:-"data_test_vc.jsonl"}   # 占位空文件

SFT_DIR=${SFT_DIR:-"${SCRIPT_DIR}/data_vc"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs_vc"}

MODEL_PATH=${MODEL_PATH:-"/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B"}

# 官方 test 集（用于评估，跟 baseline / aug 保持一致）
OFFICIAL_TEST=${OFFICIAL_TEST:-"${SCRIPT_DIR}/data/test.jsonl"}

# ---------- 训练超参（严格对齐 baseline） ----------
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-2e-5}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
EARLY_STOP_THRESHOLD=${EARLY_STOP_THRESHOLD:-0.0}

# ---------- 打印配置 ----------
echo "============================================================"
echo "[run_vc_pipeline] VC_DATA_DIR    = ${VC_DATA_DIR}"
echo "[run_vc_pipeline]   train_name   = ${TRAIN_NAME}"
echo "[run_vc_pipeline]   val_name     = ${VAL_NAME}"
echo "[run_vc_pipeline]   test_name    = ${TEST_NAME}  (占位空文件)"
echo "[run_vc_pipeline] SFT_DIR        = ${SFT_DIR}"
echo "[run_vc_pipeline] OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "[run_vc_pipeline] MODEL_PATH     = ${MODEL_PATH}"
echo "[run_vc_pipeline] OFFICIAL_TEST  = ${OFFICIAL_TEST}"
echo "------------------------------------------------------------"
echo "[run_vc_pipeline] NUM_GPUS       = ${NUM_GPUS}"
echo "[run_vc_pipeline] EPOCHS         = ${EPOCHS}"
echo "[run_vc_pipeline] BATCH_SIZE     = ${BATCH_SIZE}"
echo "[run_vc_pipeline] GRAD_ACC       = ${GRAD_ACC}"
echo "[run_vc_pipeline] LR             = ${LR}"
echo "[run_vc_pipeline] SAVE_STEPS     = ${SAVE_STEPS}"
echo "[run_vc_pipeline] SAVE_TOTAL_LIMIT      = ${SAVE_TOTAL_LIMIT}"
echo "[run_vc_pipeline] EARLY_STOP_PATIENCE   = ${EARLY_STOP_PATIENCE}"
echo "[run_vc_pipeline] EARLY_STOP_THRESHOLD  = ${EARLY_STOP_THRESHOLD}"
echo "============================================================"

# ---------- Step 0: 前置检查 ----------
if [[ ! -f "${VC_DATA_DIR}/${TRAIN_NAME}" ]]; then
    echo "[run_vc_pipeline] ERROR: train jsonl not found: ${VC_DATA_DIR}/${TRAIN_NAME}" >&2
    exit 1
fi
if [[ ! -f "${VC_DATA_DIR}/${VAL_NAME}" ]]; then
    echo "[run_vc_pipeline] ERROR: val jsonl not found: ${VC_DATA_DIR}/${VAL_NAME}" >&2
    exit 1
fi
if [[ ! -f "${OFFICIAL_TEST}" ]]; then
    echo "[run_vc_pipeline] ERROR: 官方 test 集不存在: ${OFFICIAL_TEST}" >&2
    echo "[run_vc_pipeline]        请先跑一次 baseline 的 prepare_data 生成 challenge_full_ft/data/test.jsonl" >&2
    exit 1
fi

# ---------- Step 1: 创建 test 占位空文件 ----------
echo
echo "[run_vc_pipeline][1/4] 创建 test 占位文件（prepare_data.py 强制三份都要存在）"
TEST_PATH="${VC_DATA_DIR}/${TEST_NAME}"
if [[ ! -f "${TEST_PATH}" ]]; then
    : > "${TEST_PATH}"
    echo "[run_vc_pipeline]     touched empty: ${TEST_PATH}"
else
    echo "[run_vc_pipeline]     already exists: ${TEST_PATH}"
fi

# ---------- Step 2: 转 SFT 格式 ----------
echo
echo "[run_vc_pipeline][2/4] prepare_data.py -> ${SFT_DIR}"
python "${SCRIPT_DIR}/prepare_data.py" \
    --src_dir    "${VC_DATA_DIR}" \
    --train_name "${TRAIN_NAME}" \
    --val_name   "${VAL_NAME}" \
    --test_name  "${TEST_NAME}" \
    --out_dir    "${SFT_DIR}" \
    --check_audio_exists 0

SFT_TRAIN="${SFT_DIR}/train.jsonl"
SFT_VAL="${SFT_DIR}/val.jsonl"
if [[ ! -s "${SFT_TRAIN}" || ! -s "${SFT_VAL}" ]]; then
    echo "[run_vc_pipeline] ERROR: SFT jsonl 生成失败，train/val 为空。" >&2
    exit 1
fi
echo "[run_vc_pipeline]     SFT train lines = $(wc -l < "${SFT_TRAIN}")"
echo "[run_vc_pipeline]     SFT val   lines = $(wc -l < "${SFT_VAL}")"

# ---------- Step 3: 训练 ----------
echo
echo "[run_vc_pipeline][3/4] 启动训练：${NUM_GPUS} 卡 × EPOCHS=${EPOCHS}"
NUM_GPUS="${NUM_GPUS}" \
MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${SFT_TRAIN}" \
EVAL_FILE="${SFT_VAL}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EPOCHS="${EPOCHS}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACC="${GRAD_ACC}" \
LR="${LR}" \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE}" \
EARLY_STOP_THRESHOLD="${EARLY_STOP_THRESHOLD}" \
bash "${SCRIPT_DIR}/train_full_ft.sh"

# ---------- Step 4: 评估（挑最佳 ckpt + 推理 + CER） ----------
echo
echo "[run_vc_pipeline][4/4] 评估：挑最佳 ckpt -> 推理 ${OFFICIAL_TEST} -> 计算 CER"
PRED_JSONL="${OUTPUT_DIR}/pred_test.jsonl"
WER_DIR="${OUTPUT_DIR}/wer_eval"

OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_TEST="${OFFICIAL_TEST}" \
PRED_JSONL="${PRED_JSONL}" \
WER_DIR="${WER_DIR}" \
bash "${SCRIPT_DIR}/run_eval.sh"

# ---------- Step 5: 打印最终结果 ----------
echo
echo "============================================================"
echo "[run_vc_pipeline] ★ 全流程完成，关键产物："
echo "  - 训练输出目录 : ${OUTPUT_DIR}"
echo "  - 最佳 ckpt    : $(cat "${OUTPUT_DIR}/best_ckpt.txt" 2>/dev/null || echo 'N/A')"
echo "  - 推理结果     : ${PRED_JSONL}"
echo "  - WER 目录     : ${WER_DIR}"
echo "------------------------------------------------------------"
if [[ -f "${WER_DIR}/result.wer" ]]; then
    echo "[整体 CER] (${WER_DIR}/result.wer):"
    cat "${WER_DIR}/result.wer"
else
    echo "[run_vc_pipeline] WARN: 未找到 ${WER_DIR}/result.wer" >&2
fi
echo "------------------------------------------------------------"
if [[ -f "${WER_DIR}/by_dialect_summary.txt" ]]; then
    echo "[按方言 CER] (${WER_DIR}/by_dialect_summary.txt):"
    cat "${WER_DIR}/by_dialect_summary.txt"
else
    echo "[run_vc_pipeline] WARN: 未找到 ${WER_DIR}/by_dialect_summary.txt" >&2
fi
echo "------------------------------------------------------------"
if [[ -f "${WER_DIR}/dialect_accuracy.txt" ]]; then
    echo "[方言分类准确率] (${WER_DIR}/dialect_accuracy.txt):"
    cat "${WER_DIR}/dialect_accuracy.txt"
fi
echo "============================================================"

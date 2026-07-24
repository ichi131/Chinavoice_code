#!/usr/bin/env bash
# =============================================================================
# run_mix_pipeline.sh
# -----------------------------------------------------------------------------
# 一键跑通「VC_data_v2 + basic_change_data 混合数据」的全流程：
#   Step 1  合并两份 jsonl -> mix_data/data_{train,val}_mix.jsonl（原始去重）
#   Step 2  prepare_data.py：转 SFT 格式 -> data_mix/
#   Step 3  train_full_ft.sh：8 卡训练，超参严格对齐 baseline -> outputs_mix/
#   Step 4  run_eval.sh：挑最佳 ckpt -> 8 卡推理官方 test -> 计算 CER
#   Step 5  打印最终结果
#
# 使用：
#   bash challenge_full_ft/run_mix_pipeline.sh 2>&1 | tee run_mix_pipeline.log
# =============================================================================

set -euo pipefail

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VC_DIR=${VC_DIR:-"${ROOT_DIR}/VC_data_v2"}
BASIC_DIR=${BASIC_DIR:-"${ROOT_DIR}/basic_change_data"}
MIX_DIR=${MIX_DIR:-"${ROOT_DIR}/mix_data"}                    # 合并后原始格式 jsonl
SFT_DIR=${SFT_DIR:-"${SCRIPT_DIR}/data_mix"}                  # SFT 格式产物
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs_mix"}         # 训练输出

MODEL_PATH=${MODEL_PATH:-"/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B"}
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
echo "[run_mix_pipeline] VC_DIR         = ${VC_DIR}"
echo "[run_mix_pipeline] BASIC_DIR      = ${BASIC_DIR}"
echo "[run_mix_pipeline] MIX_DIR        = ${MIX_DIR}"
echo "[run_mix_pipeline] SFT_DIR        = ${SFT_DIR}"
echo "[run_mix_pipeline] OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "[run_mix_pipeline] MODEL_PATH     = ${MODEL_PATH}"
echo "[run_mix_pipeline] OFFICIAL_TEST  = ${OFFICIAL_TEST}"
echo "------------------------------------------------------------"
echo "[run_mix_pipeline] NUM_GPUS       = ${NUM_GPUS}"
echo "[run_mix_pipeline] EPOCHS         = ${EPOCHS}"
echo "[run_mix_pipeline] BATCH_SIZE     = ${BATCH_SIZE}"
echo "[run_mix_pipeline] GRAD_ACC       = ${GRAD_ACC}"
echo "[run_mix_pipeline] LR             = ${LR}"
echo "[run_mix_pipeline] SAVE_STEPS     = ${SAVE_STEPS}"
echo "[run_mix_pipeline] SAVE_TOTAL_LIMIT      = ${SAVE_TOTAL_LIMIT}"
echo "[run_mix_pipeline] EARLY_STOP_PATIENCE   = ${EARLY_STOP_PATIENCE}"
echo "[run_mix_pipeline] EARLY_STOP_THRESHOLD  = ${EARLY_STOP_THRESHOLD}"
echo "============================================================"

# ---------- 前置检查 ----------
for f in \
    "${VC_DIR}/data_train_vc.jsonl" \
    "${VC_DIR}/data_val_vc.jsonl" \
    "${BASIC_DIR}/data_train_aug.jsonl" \
    "${BASIC_DIR}/data_val_aug.jsonl" \
    "${OFFICIAL_TEST}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[run_mix_pipeline] ERROR: file not found: ${f}" >&2
        exit 1
    fi
done

# ---------- Step 1: 合并两份 jsonl（原始去重） ----------
echo
echo "[run_mix_pipeline][1/4] merge VC + basic -> ${MIX_DIR}"
python "${SCRIPT_DIR}/merge_mix_data.py" \
    --vc_dir    "${VC_DIR}" \
    --basic_dir "${BASIC_DIR}" \
    --out_dir   "${MIX_DIR}"

MIX_TRAIN="${MIX_DIR}/data_train_mix.jsonl"
MIX_VAL="${MIX_DIR}/data_val_mix.jsonl"
MIX_TEST="${MIX_DIR}/data_test_mix.jsonl"
if [[ ! -s "${MIX_TRAIN}" || ! -s "${MIX_VAL}" ]]; then
    echo "[run_mix_pipeline] ERROR: merged jsonl empty. train=${MIX_TRAIN} val=${MIX_VAL}" >&2
    exit 1
fi
echo "[run_mix_pipeline]     merged train lines = $(wc -l < "${MIX_TRAIN}")"
echo "[run_mix_pipeline]     merged val   lines = $(wc -l < "${MIX_VAL}")"

# ---------- Step 2: 转 SFT 格式 ----------
echo
echo "[run_mix_pipeline][2/4] prepare_data.py -> ${SFT_DIR}"
python "${SCRIPT_DIR}/prepare_data.py" \
    --src_dir    "${MIX_DIR}" \
    --train_name "data_train_mix.jsonl" \
    --val_name   "data_val_mix.jsonl" \
    --test_name  "data_test_mix.jsonl" \
    --out_dir    "${SFT_DIR}" \
    --check_audio_exists 0

SFT_TRAIN="${SFT_DIR}/train.jsonl"
SFT_VAL="${SFT_DIR}/val.jsonl"
if [[ ! -s "${SFT_TRAIN}" || ! -s "${SFT_VAL}" ]]; then
    echo "[run_mix_pipeline] ERROR: SFT jsonl generation failed." >&2
    exit 1
fi
echo "[run_mix_pipeline]     SFT train lines = $(wc -l < "${SFT_TRAIN}")"
echo "[run_mix_pipeline]     SFT val   lines = $(wc -l < "${SFT_VAL}")"

# ---------- Step 3: 训练 ----------
echo
echo "[run_mix_pipeline][3/4] 启动训练：${NUM_GPUS} 卡 × EPOCHS=${EPOCHS}"
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

# ---------- Step 4: 评估 ----------
echo
echo "[run_mix_pipeline][4/4] 评估：挑最佳 ckpt -> 推理 ${OFFICIAL_TEST} -> 计算 CER"
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
echo "[run_mix_pipeline] ★ 全流程完成，关键产物："
echo "  - 合并后原始 jsonl : ${MIX_DIR}"
echo "  - SFT 格式         : ${SFT_DIR}"
echo "  - 训练输出         : ${OUTPUT_DIR}"
echo "  - 最佳 ckpt        : $(cat "${OUTPUT_DIR}/best_ckpt.txt" 2>/dev/null || echo 'N/A')"
echo "  - 推理结果         : ${PRED_JSONL}"
echo "  - WER 目录         : ${WER_DIR}"
echo "------------------------------------------------------------"
if [[ -f "${WER_DIR}/result.wer" ]]; then
    echo "[整体 CER] (${WER_DIR}/result.wer):"
    cat "${WER_DIR}/result.wer"
else
    echo "[run_mix_pipeline] WARN: 未找到 ${WER_DIR}/result.wer" >&2
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

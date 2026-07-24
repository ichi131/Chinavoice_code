#!/usr/bin/env bash
# =============================================================================
# run_specialist_pipeline.sh
# -----------------------------------------------------------------------------
# 单方言（specialist）Qwen3-ASR 微调 pipeline。
#
# 输入：ACCENT 环境变量 or 首个位置参数（例如 cantonese / nanjing / ...）
#
# 流程（可断点续跑，中间产物已存在即跳过）：
#   [1] 数据准备          -> data_specialist/{accent}/{train,val,test}.jsonl
#   [2] 训练              -> outputs_specialist/{accent}/
#   [3] 推理 + CER 评估   -> outputs_specialist/{accent}/pred_test.jsonl
#                          + outputs_specialist/{accent}/wer_eval/
#
# 用法：
#   ACCENT=cantonese bash challenge_full_ft/run_specialist_pipeline.sh
#   或：
#   bash challenge_full_ft/run_specialist_pipeline.sh cantonese
#
# 常用环境变量：
#   ACCENT             必填。目标方言
#   FORCE_REBUILD      1/0(默认)  强制重建数据
#   FORCE_TRAIN        1/0(默认)  强制重训
#   FORCE_EVAL         1/0(默认)  强制重推理
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

# ---------- 方言参数 ----------
ACCENT=${ACCENT:-"${1:-}"}
if [[ -z "${ACCENT}" ]]; then
    echo "[run_specialist_pipeline] ERROR: 请通过 ACCENT= 或第一个位置参数指定方言" >&2
    exit 1
fi

ALL_ACCENTS=(anhui cantonese changsha chaoshan dongbei henan kejia minnan
             nanchang nanjing shan1xi shan3xi shandong sichuan wuhan wuyu)
_valid=0
for a in "${ALL_ACCENTS[@]}"; do
    if [[ "${a}" == "${ACCENT}" ]]; then _valid=1; break; fi
done
if [[ "${_valid}" -ne 1 ]]; then
    echo "[run_specialist_pipeline] ERROR: 未知方言 '${ACCENT}'，合法值：${ALL_ACCENTS[*]}" >&2
    exit 1
fi

# ---------- 原始数据源 ----------
VC_V2_TRAIN=${VC_V2_TRAIN:-"${ROOT_DIR}/VC_data_v2/data_train_vc.jsonl"}
VC_V2_VAL=${VC_V2_VAL:-"${ROOT_DIR}/VC_data_v2/data_val_vc.jsonl"}
RAW_TRAIN=${RAW_TRAIN:-"${SCRIPT_DIR}/data/train.jsonl"}
RAW_VAL=${RAW_VAL:-"${SCRIPT_DIR}/data/val.jsonl"}
RAW_TEST=${RAW_TEST:-"${SCRIPT_DIR}/data/test.jsonl"}

# ---------- 产物目录 ----------
DATA_ROOT=${DATA_ROOT:-"${SCRIPT_DIR}/data_specialist"}
OUT_ROOT=${OUT_ROOT:-"${SCRIPT_DIR}/outputs_specialist"}
DATA_DIR="${DATA_ROOT}/${ACCENT}"
OUT_DIR="${OUT_ROOT}/${ACCENT}"

BASE_MODEL=${BASE_MODEL:-"/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B"}

# ---------- 训练超参（对齐 wuyu Stage 2，EPOCHS 上调到 10，让早停生效） ----------
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER=${LR_SCHEDULER:-"cosine"}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
EARLY_STOP_THRESHOLD=${EARLY_STOP_THRESHOLD:-0.0}
BF16_VAL=${BF16_VAL:-1}

# ---------- 控制变量 ----------
FORCE_REBUILD=${FORCE_REBUILD:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_EVAL=${FORCE_EVAL:-0}

# ---------- 打印配置 ----------
echo "============================================================"
echo "[run_specialist_pipeline] ACCENT              = ${ACCENT}"
echo "[run_specialist_pipeline] VC_V2_TRAIN         = ${VC_V2_TRAIN}"
echo "[run_specialist_pipeline] VC_V2_VAL           = ${VC_V2_VAL}"
echo "[run_specialist_pipeline] RAW_TRAIN           = ${RAW_TRAIN}"
echo "[run_specialist_pipeline] RAW_VAL             = ${RAW_VAL}"
echo "[run_specialist_pipeline] RAW_TEST            = ${RAW_TEST}"
echo "[run_specialist_pipeline] DATA_DIR            = ${DATA_DIR}"
echo "[run_specialist_pipeline] OUT_DIR             = ${OUT_DIR}"
echo "[run_specialist_pipeline] BASE_MODEL          = ${BASE_MODEL}"
echo "[run_specialist_pipeline] NUM_GPUS            = ${NUM_GPUS}"
echo "[run_specialist_pipeline] EPOCHS              = ${EPOCHS}"
echo "[run_specialist_pipeline] BATCH_SIZE          = ${BATCH_SIZE}"
echo "[run_specialist_pipeline] GRAD_ACC            = ${GRAD_ACC}"
echo "[run_specialist_pipeline] LR                  = ${LR}"
echo "[run_specialist_pipeline] SAVE_STEPS          = ${SAVE_STEPS}"
echo "[run_specialist_pipeline] EARLY_STOP_PATIENCE = ${EARLY_STOP_PATIENCE}"
echo "[run_specialist_pipeline] FORCE_REBUILD       = ${FORCE_REBUILD}"
echo "[run_specialist_pipeline] FORCE_TRAIN         = ${FORCE_TRAIN}"
echo "[run_specialist_pipeline] FORCE_EVAL          = ${FORCE_EVAL}"
echo "============================================================"

# ---------- 前置检查 ----------
for f in "${VC_V2_TRAIN}" "${VC_V2_VAL}" "${RAW_TRAIN}" "${RAW_VAL}" "${RAW_TEST}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[run_specialist_pipeline] ERROR: 输入不存在: ${f}" >&2
        exit 1
    fi
done

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

FORCE_FLAG=""
if [[ "${FORCE_REBUILD}" == "1" ]]; then
    FORCE_FLAG="--force"
fi

# =============================================================================
# [Step 1] 数据准备
# =============================================================================
echo
echo "[run_specialist_pipeline:${ACCENT}][Step 1/3] 数据准备 -> ${DATA_DIR}"
python "${SCRIPT_DIR}/prepare_specialist_data.py" \
    --accent "${ACCENT}" \
    --vc_train "${VC_V2_TRAIN}" \
    --vc_val   "${VC_V2_VAL}" \
    --raw_train "${RAW_TRAIN}" \
    --raw_val   "${RAW_VAL}" \
    --raw_test  "${RAW_TEST}" \
    --out_root  "${DATA_ROOT}" \
    --check_audio_exists 0 \
    ${FORCE_FLAG}

TRAIN_FILE="${DATA_DIR}/train.jsonl"
VAL_FILE="${DATA_DIR}/val.jsonl"
TEST_FILE="${DATA_DIR}/test.jsonl"
for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
    if [[ ! -s "${f}" ]]; then
        echo "[run_specialist_pipeline:${ACCENT}] ERROR: 数据文件为空: ${f}" >&2
        exit 1
    fi
done
echo "[run_specialist_pipeline:${ACCENT}] train = $(wc -l < "${TRAIN_FILE}")"
echo "[run_specialist_pipeline:${ACCENT}] val   = $(wc -l < "${VAL_FILE}")"
echo "[run_specialist_pipeline:${ACCENT}] test  = $(wc -l < "${TEST_FILE}")"

# =============================================================================
# [Step 2] 训练
# =============================================================================
has_valid_best_ckpt() {
    local d="$1"
    local f="${d}/best_ckpt.txt"
    [[ -s "${f}" ]] || return 1
    local ckpt; ckpt="$(cat "${f}" | tr -d '[:space:]')"
    [[ -n "${ckpt}" && -f "${ckpt}/config.json" ]] || return 1
    return 0
}

echo
echo "[run_specialist_pipeline:${ACCENT}][Step 2/3] 训练 -> ${OUT_DIR}"
if has_valid_best_ckpt "${OUT_DIR}" && [[ "${FORCE_TRAIN}" != "1" ]]; then
    echo "[run_specialist_pipeline:${ACCENT}] 已存在有效 best_ckpt，SKIP"
    echo "[run_specialist_pipeline:${ACCENT}]   best = $(cat "${OUT_DIR}/best_ckpt.txt")"
else
    NUM_GPUS="${NUM_GPUS}" \
    MODEL_PATH="${BASE_MODEL}" \
    TRAIN_FILE="${TRAIN_FILE}" \
    EVAL_FILE="${VAL_FILE}" \
    OUTPUT_DIR="${OUT_DIR}" \
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
    BF16="${BF16_VAL}" \
    bash "${SCRIPT_DIR}/train_full_ft.sh"

    if ! has_valid_best_ckpt "${OUT_DIR}"; then
        echo "[run_specialist_pipeline:${ACCENT}] ERROR: 训练结束但 best_ckpt.txt 无效" >&2
        exit 1
    fi
    echo "[run_specialist_pipeline:${ACCENT}] 训练完成，best = $(cat "${OUT_DIR}/best_ckpt.txt")"
fi

# =============================================================================
# [Step 3] 推理 + CER
# =============================================================================
PRED_JSONL="${OUT_DIR}/pred_test.jsonl"
WER_DIR="${OUT_DIR}/wer_eval"

echo
echo "[run_specialist_pipeline:${ACCENT}][Step 3/3] 推理 + CER -> ${WER_DIR}"

# 判断是否已有完整评估产物
if [[ -s "${WER_DIR}/result.wer" && -s "${PRED_JSONL}" && "${FORCE_EVAL}" != "1" ]]; then
    echo "[run_specialist_pipeline:${ACCENT}] wer_eval/result.wer 已存在，SKIP"
else
    # 若已有 pred 但缺 WER：仅重跑 WER
    if [[ -s "${PRED_JSONL}" && "${FORCE_EVAL}" != "1" ]]; then
        echo "[run_specialist_pipeline:${ACCENT}] pred 已存在，仅重算 CER"
        EVAL_TOOL_SH="/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/ChinaVoices-Challenge/eval/eval_jsonl_with_wer_tools.sh"
        mkdir -p "${WER_DIR}"
        bash "${EVAL_TOOL_SH}" \
            --pred_jsonl "${PRED_JSONL}" \
            --output_dir "${WER_DIR}" \
            --apply_t2s  1 \
            --by_dialect 1
    else
        OUTPUT_DIR="${OUT_DIR}" \
        DATA_TEST="${TEST_FILE}" \
        PRED_JSONL="${PRED_JSONL}" \
        WER_DIR="${WER_DIR}" \
        NUM_GPUS="${NUM_GPUS}" \
        bash "${SCRIPT_DIR}/run_eval.sh"
    fi
fi

echo
echo "============================================================"
echo "[run_specialist_pipeline:${ACCENT}] ★ 完成"
echo "  best_ckpt  : $(cat "${OUT_DIR}/best_ckpt.txt")"
echo "  pred       : ${PRED_JSONL}"
echo "  wer file   : ${WER_DIR}/result.wer"
echo "============================================================"

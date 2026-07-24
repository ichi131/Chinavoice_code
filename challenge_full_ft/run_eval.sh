#!/usr/bin/env bash
# =============================================================================
# run_eval.sh
# -----------------------------------------------------------------------------
# 一键完成：挑最佳 ckpt → 在 test 集上批量推理 → 调用 ChinaVoices-Challenge 的
# eval_jsonl_with_wer_tools.sh 计算整体 CER 与按方言 CER。
#
# 所有关键变量都可通过环境变量覆盖：
#   OUTPUT_DIR       训练输出根目录（默认 ./outputs）
#   DATA_TEST        prepare_data.py 产出的 test.jsonl
#                    （默认 ./data/test.jsonl）
#   MODEL_CKPT       手动指定推理 ckpt；留空则调用 pick_best_ckpt.py 挑选
#   EVAL_TOOL_SH     ChinaVoices-Challenge 的 eval 入口 shell
#   PRED_JSONL       推理结果 JSONL 保存路径（默认 OUTPUT_DIR/pred_test.jsonl）
#   WER_DIR          WER 评估输出目录（默认 OUTPUT_DIR/wer_eval）
#   BATCH_SIZE / MAX_TOKENS / DEVICE_MAP  推理相关
#   APPLY_T2S / BY_DIALECT                eval 工具参数
#
# ⚠️ 仅在实验机上运行。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---- 路径 ----
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs"}
DATA_TEST=${DATA_TEST:-"${SCRIPT_DIR}/data/test.jsonl"}
MODEL_CKPT=${MODEL_CKPT:-""}
EVAL_TOOL_SH=${EVAL_TOOL_SH:-"/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/ChinaVoices-Challenge/eval/eval_jsonl_with_wer_tools.sh"}
PRED_JSONL=${PRED_JSONL:-"${OUTPUT_DIR}/pred_test.jsonl"}
WER_DIR=${WER_DIR:-"${OUTPUT_DIR}/wer_eval"}

# ---- 推理超参 ----
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-512}
# 单卡时使用的 device_map（多卡时子进程内部也用 cuda:0）。
# 切勿设为 "auto"：它会把 Qwen3-ASR 拆到多卡，引发
# "Expected all tensors to be on the same device" 报错。
DEVICE_MAP=${DEVICE_MAP:-"cuda:0"}
# 并行推理的进程数：默认自动检测可用 GPU 数。
# =1 时走单卡 infer_test.py；>=2 时走 run_infer_multi_gpu.sh。
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-1}

# ---- eval 参数 ----
APPLY_T2S=${APPLY_T2S:-1}
BY_DIALECT=${BY_DIALECT:-1}

# ---- 挑最佳 ckpt 参数 ----
METRIC=${METRIC:-"eval_loss"}
GREATER_IS_BETTER=${GREATER_IS_BETTER:-0}

mkdir -p "${OUTPUT_DIR}" "${WER_DIR}"

echo "============================================================"
echo "[run_eval] OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "[run_eval] DATA_TEST    = ${DATA_TEST}"
echo "[run_eval] EVAL_TOOL_SH = ${EVAL_TOOL_SH}"
echo "[run_eval] PRED_JSONL   = ${PRED_JSONL}"
echo "[run_eval] WER_DIR      = ${WER_DIR}"
echo "[run_eval] NUM_GPUS     = ${NUM_GPUS}"
echo "============================================================"

# ---- Step 1: 挑最佳 ckpt ----
if [[ -z "${MODEL_CKPT}" ]]; then
    echo "[run_eval][1/3] pick best ckpt from ${OUTPUT_DIR} ..."
    MODEL_CKPT="$(python "${SCRIPT_DIR}/pick_best_ckpt.py" \
        --output_dir "${OUTPUT_DIR}" \
        --metric "${METRIC}" \
        --greater_is_better "${GREATER_IS_BETTER}")"
    if [[ -z "${MODEL_CKPT}" ]]; then
        echo "[run_eval] ERROR: pick_best_ckpt.py returned empty path." >&2
        exit 1
    fi
else
    echo "[run_eval][1/3] use user-specified MODEL_CKPT."
fi
echo "[run_eval]     MODEL_CKPT = ${MODEL_CKPT}"

if [[ ! -f "${MODEL_CKPT}/config.json" ]]; then
    echo "[run_eval] ERROR: ${MODEL_CKPT}/config.json not found." >&2
    echo "[run_eval]        请确认该 ckpt 是完整可推理目录（含 config.json 等 HF 文件）。" >&2
    exit 1
fi

# ---- Step 2: 推理 test 集（自动选 单卡 / 多卡并行） ----
if [[ "${NUM_GPUS}" -ge 2 ]]; then
    echo "[run_eval][2/3] multi-GPU (${NUM_GPUS} gpus) infer on ${DATA_TEST} -> ${PRED_JSONL}"
    NUM_GPUS="${NUM_GPUS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    DEVICE_MAP="${DEVICE_MAP}" \
    MODEL_CKPT="${MODEL_CKPT}" \
    DATA_TEST="${DATA_TEST}" \
    PRED_JSONL="${PRED_JSONL}" \
    bash "${SCRIPT_DIR}/run_infer_multi_gpu.sh"
else
    echo "[run_eval][2/3] single-GPU infer on ${DATA_TEST} -> ${PRED_JSONL}"
    python "${SCRIPT_DIR}/infer_test.py" \
        --model      "${MODEL_CKPT}" \
        --data       "${DATA_TEST}" \
        --output     "${PRED_JSONL}" \
        --batch-size "${BATCH_SIZE}" \
        --max-tokens "${MAX_TOKENS}" \
        --device-map "${DEVICE_MAP}"
fi

# ---- Step 3: 调用官方 CER 评估 ----
if [[ ! -f "${EVAL_TOOL_SH}" ]]; then
    echo "[run_eval] ERROR: EVAL_TOOL_SH not found: ${EVAL_TOOL_SH}" >&2
    exit 1
fi

echo "[run_eval][3/3] compute CER via ${EVAL_TOOL_SH}"
bash "${EVAL_TOOL_SH}" \
    --pred_jsonl "${PRED_JSONL}" \
    --output_dir "${WER_DIR}" \
    --apply_t2s  "${APPLY_T2S}" \
    --by_dialect "${BY_DIALECT}"

echo "============================================================"
echo "[run_eval] done."
echo "  overall CER      : ${WER_DIR}/result.wer"
echo "  by-dialect CER   : ${WER_DIR}/by_dialect_summary.txt"
echo "  dialect accuracy : ${WER_DIR}/dialect_accuracy.txt"
echo "============================================================"

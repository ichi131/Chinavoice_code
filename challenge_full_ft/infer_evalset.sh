#!/usr/bin/env bash
# =============================================================================
# infer_evalset.sh
# -----------------------------------------------------------------------------
# 对比赛官方 evaluation_set（无参考文本）做 8 卡并行推理并输出提交产物。
#
# 三步：
#   Step 1  wav.scp -> SFT 格式 jsonl（infer_test.py 认识的输入）
#   Step 2  run_infer_multi_gpu.sh: 8 卡并行推理，合并成 pred_eval.jsonl
#   Step 3  postprocess -> result.jsonl / result.trn
#
# 用法（默认参数已按用户需求写好）：
#   bash challenge_full_ft/infer_evalset.sh 2>&1 | tee infer_evalset.log
#
# 覆盖示例：
#   MODEL_CKPT=/other/ckpt EVAL_SCP=/other/wav.scp OUT_DIR=/other/dir \
#     bash challenge_full_ft/infer_evalset.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# ---- 用户提供的默认路径 ----
MODEL_CKPT=${MODEL_CKPT:-"/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/outputs_vc_v2/checkpoint-500"}
EVAL_SCP=${EVAL_SCP:-"/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/data/chinavoices_challenge/evaluation_set/wav.scp"}
OUT_DIR=${OUT_DIR:-"/mnt/geminihzceph/user_johannapeng/challenge_model/infer_data"}

# ---- 推理超参 ----
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-512}
CHECK_AUDIO=${CHECK_AUDIO:-0}    # 1 时逐条校验 wav 存在，10w+ 条会较慢

# ---- 中间/最终产物 ----
INFER_INPUT_JSONL="${OUT_DIR}/eval_input.jsonl"
PRED_JSONL="${OUT_DIR}/pred_eval.jsonl"
POST_DIR="${OUT_DIR}"
LOG_DIR="${OUT_DIR}/infer_logs"

echo "============================================================"
echo "[infer_evalset] MODEL_CKPT   = ${MODEL_CKPT}"
echo "[infer_evalset] EVAL_SCP     = ${EVAL_SCP}"
echo "[infer_evalset] OUT_DIR      = ${OUT_DIR}"
echo "[infer_evalset] NUM_GPUS     = ${NUM_GPUS}"
echo "[infer_evalset] BATCH_SIZE   = ${BATCH_SIZE}"
echo "[infer_evalset] MAX_TOKENS   = ${MAX_TOKENS}"
echo "[infer_evalset] CHECK_AUDIO  = ${CHECK_AUDIO}"
echo "[infer_evalset] INFER_INPUT  = ${INFER_INPUT_JSONL}"
echo "[infer_evalset] PRED_JSONL   = ${PRED_JSONL}"
echo "[infer_evalset] LOG_DIR      = ${LOG_DIR}"
echo "============================================================"

# ---- 前置检查 ----
if [[ ! -d "${MODEL_CKPT}" ]]; then
    echo "[infer_evalset][ERROR] MODEL_CKPT 目录不存在: ${MODEL_CKPT}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_CKPT}/config.json" ]]; then
    echo "[infer_evalset][ERROR] MODEL_CKPT 目录缺少 config.json: ${MODEL_CKPT}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_CKPT}/preprocessor_config.json" ]]; then
    echo "[infer_evalset][WARN] ${MODEL_CKPT} 缺少 preprocessor_config.json，" >&2
    echo "                     若 AutoProcessor 加载失败可先运行 fix_ckpt_processor.py" >&2
fi
if [[ ! -f "${EVAL_SCP}" ]]; then
    echo "[infer_evalset][ERROR] EVAL_SCP 不存在: ${EVAL_SCP}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# ---- Step 1: wav.scp -> jsonl ----
echo
echo "[infer_evalset][1/3] wav.scp -> ${INFER_INPUT_JSONL}"
python "${SCRIPT_DIR}/scp_to_infer_jsonl.py" \
    --scp "${EVAL_SCP}" \
    --out "${INFER_INPUT_JSONL}" \
    --check_audio "${CHECK_AUDIO}"

if [[ ! -s "${INFER_INPUT_JSONL}" ]]; then
    echo "[infer_evalset][ERROR] 转换出的 jsonl 为空: ${INFER_INPUT_JSONL}" >&2
    exit 1
fi
N_INPUT=$(wc -l < "${INFER_INPUT_JSONL}")
echo "[infer_evalset]     conv done: ${N_INPUT} lines"

# ---- Step 2: 8 卡并行推理 ----
echo
echo "[infer_evalset][2/3] 8 卡并行推理 -> ${PRED_JSONL}"
MODEL_CKPT="${MODEL_CKPT}" \
DATA_TEST="${INFER_INPUT_JSONL}" \
PRED_JSONL="${PRED_JSONL}" \
NUM_GPUS="${NUM_GPUS}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_TOKENS="${MAX_TOKENS}" \
LOG_DIR="${LOG_DIR}" \
CLEAN_SHARDS=1 \
CLEAN_LOGS=0 \
bash "${SCRIPT_DIR}/run_infer_multi_gpu.sh"

if [[ ! -s "${PRED_JSONL}" ]]; then
    echo "[infer_evalset][ERROR] 推理结果为空: ${PRED_JSONL}" >&2
    exit 1
fi
N_PRED=$(wc -l < "${PRED_JSONL}")
echo "[infer_evalset]     pred done: ${N_PRED} lines"

if [[ "${N_PRED}" != "${N_INPUT}" ]]; then
    echo "[infer_evalset][WARN] 输入行数=${N_INPUT} 与推理行数=${N_PRED} 不一致，请检查日志" >&2
fi

# ---- Step 3: 后处理 ----
echo
echo "[infer_evalset][3/3] postprocess -> ${POST_DIR}"
python "${SCRIPT_DIR}/postprocess_pred_evalset.py" \
    --pred    "${PRED_JSONL}" \
    --out_dir "${POST_DIR}"

echo
echo "============================================================"
echo "[infer_evalset] ★ 完成，产物："
echo "  - 输入 jsonl (from scp)    : ${INFER_INPUT_JSONL}"
echo "  - 推理原始 jsonl (合并后)  : ${PRED_JSONL}"
echo "  - 精简 jsonl               : ${POST_DIR}/result.jsonl"
echo "  - Kaldi 风格 trn           : ${POST_DIR}/result.trn"
echo "  - 推理进程日志目录         : ${LOG_DIR}"
echo "============================================================"

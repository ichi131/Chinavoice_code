#!/usr/bin/env bash
# =============================================================================
# challenge_full_ft/run_infer_confidence_multi_gpu.sh
# -----------------------------------------------------------------------------
# 方案 A（带置信度推理）8 卡并行封装脚本。与 run_infer_multi_gpu.sh **完全平行**
# 的一份克隆，唯一区别是调用 infer_test_with_confidence.py 而非 infer_test.py，
# 因此产出的 JSONL 会多出 dialect_conf / dialect_logprob / text_avg_logprob
# 等置信度字段，其余行为（stride 切分、合并分片、CLEAN_SHARDS / CLEAN_LOGS 清理
# 开关）完全一致。
#
# 用法：
#   MODEL_CKPT=/abs/path/to/checkpoint-XXXX \
#   DATA_TEST=challenge_full_ft/data/test.jsonl \
#   PRED_JSONL=challenge_full_ft/outputs/pred_test_conf.jsonl \
#   NUM_GPUS=8 BATCH_SIZE=16 MAX_TOKENS=512 \
#   bash challenge_full_ft/run_infer_confidence_multi_gpu.sh
#
# 注意：BATCH_SIZE 建议比原来（32）**减半到 16**，因为带 output_scores 会额外
# 保留 T_new * vocab_size 的 float 张量，显存占用翻倍甚至更多。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

MODEL_CKPT=${MODEL_CKPT:?"[run_infer_confidence] MODEL_CKPT 必填：checkpoint 目录绝对路径"}
DATA_TEST=${DATA_TEST:-"${SCRIPT_DIR}/data/test.jsonl"}
PRED_JSONL=${PRED_JSONL:-"${SCRIPT_DIR}/outputs/pred_test_conf.jsonl"}

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_TOKENS=${MAX_TOKENS:-512}
DEVICE_MAP=${DEVICE_MAP:-"cuda:0"}

LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/infer_logs"}
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${PRED_JSONL}")"

CLEAN_SHARDS=${CLEAN_SHARDS:-1}
CLEAN_LOGS=${CLEAN_LOGS:-0}

echo "[run_infer_conf] REPO_ROOT   = ${REPO_ROOT}"
echo "[run_infer_conf] MODEL_CKPT  = ${MODEL_CKPT}"
echo "[run_infer_conf] DATA_TEST   = ${DATA_TEST}"
echo "[run_infer_conf] PRED_JSONL  = ${PRED_JSONL}"
echo "[run_infer_conf] NUM_GPUS    = ${NUM_GPUS}"
echo "[run_infer_conf] BATCH_SIZE  = ${BATCH_SIZE}"
echo "[run_infer_conf] MAX_TOKENS  = ${MAX_TOKENS}"
echo "[run_infer_conf] DEVICE_MAP  = ${DEVICE_MAP}"
echo "[run_infer_conf] LOG_DIR     = ${LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}/Qwen3-ASR:${PYTHONPATH:-}"

OUTPUT_BASE="${PRED_JSONL%.*}"
OUTPUT_EXT="${PRED_JSONL##*.}"
if [[ "${OUTPUT_BASE}" == "${PRED_JSONL}" ]]; then
    OUTPUT_EXT=""
fi

PIDS=()
for ((rank=0; rank<NUM_GPUS; rank++)); do
    LOG="${LOG_DIR}/infer_conf_rank${rank}.log"
    echo "[run_infer_conf] 启动 rank=${rank} -> GPU ${rank}, 日志: ${LOG}"
    CUDA_VISIBLE_DEVICES=${rank} \
    python "${SCRIPT_DIR}/infer_test_with_confidence.py" \
        --model       "${MODEL_CKPT}" \
        --data        "${DATA_TEST}" \
        --output      "${PRED_JSONL}" \
        --batch-size  "${BATCH_SIZE}" \
        --max-tokens  "${MAX_TOKENS}" \
        --device-map  "${DEVICE_MAP}" \
        --rank        "${rank}" \
        --world-size  "${NUM_GPUS}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

fail=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "[run_infer_conf][ERROR] 进程 ${pid} 失败"
        fail=1
    fi
done
if [[ ${fail} -ne 0 ]]; then
    echo "[run_infer_conf][ERROR] 有进程失败，请查看 ${LOG_DIR}/ 下的日志"
    exit 1
fi

echo "[run_infer_conf] 合并分片到 ${PRED_JSONL}"
> "${PRED_JSONL}"
for ((rank=0; rank<NUM_GPUS; rank++)); do
    if [[ -n "${OUTPUT_EXT}" ]]; then
        SHARD="${OUTPUT_BASE}.rank${rank}.${OUTPUT_EXT}"
    else
        SHARD="${OUTPUT_BASE}.rank${rank}"
    fi
    if [[ -f "${SHARD}" ]]; then
        cat "${SHARD}" >> "${PRED_JSONL}"
    else
        echo "[run_infer_conf][WARN] 找不到分片: ${SHARD}"
    fi
done

TOTAL=$(wc -l < "${PRED_JSONL}")
echo "[run_infer_conf][DONE] 合并完成: ${PRED_JSONL}  共 ${TOTAL} 行"

if [[ "${TOTAL}" -gt 0 ]]; then
    if [[ "${CLEAN_SHARDS}" == "1" ]]; then
        echo "[run_infer_conf] CLEAN_SHARDS=1，删除分片文件..."
        for ((rank=0; rank<NUM_GPUS; rank++)); do
            if [[ -n "${OUTPUT_EXT}" ]]; then
                SHARD="${OUTPUT_BASE}.rank${rank}.${OUTPUT_EXT}"
            else
                SHARD="${OUTPUT_BASE}.rank${rank}"
            fi
            [[ -f "${SHARD}" ]] && rm -f "${SHARD}"
        done
    fi
    if [[ "${CLEAN_LOGS}" == "1" ]]; then
        rm -f "${LOG_DIR}"/infer_conf_rank*.log
    fi
fi

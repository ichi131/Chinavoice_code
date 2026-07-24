#!/usr/bin/env bash
# =============================================================================
# challenge_full_ft/run_infer_multi_gpu.sh
# -----------------------------------------------------------------------------
# 8 卡（可配置）并行推理封装脚本。套路与官方 Qwen3-ASR/run_infer_multi_gpu.sh
# 一致：
#   1) 每张 GPU 起一个 infer_test.py 子进程；
#   2) 各进程通过 CUDA_VISIBLE_DEVICES=rank 独占一张卡，进程内用 device_map="cuda:0"
#      单卡加载模型（**不要用 device_map="auto"**，会拆碎模型跨卡触发张量冲突）；
#   3) 每个进程只处理 idx % world_size == rank 的样本（stride 切分，负载均衡）；
#   4) 各自写 pred_test.rank{N}.jsonl，全部结束后 cat 合并成最终 pred_test.jsonl。
#
# 用法：
#   MODEL_CKPT=/abs/path/to/checkpoint-XXXX \
#   DATA_TEST=challenge_full_ft/data/test.jsonl \
#   PRED_JSONL=challenge_full_ft/outputs/pred_test.jsonl \
#   NUM_GPUS=8 BATCH_SIZE=32 MAX_TOKENS=512 \
#   bash challenge_full_ft/run_infer_multi_gpu.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

# ---- 必填 / 可选 参数 ----
MODEL_CKPT=${MODEL_CKPT:?"[run_infer_multi_gpu] MODEL_CKPT 必填：checkpoint 目录绝对路径"}
DATA_TEST=${DATA_TEST:-"${SCRIPT_DIR}/data/test.jsonl"}
PRED_JSONL=${PRED_JSONL:-"${SCRIPT_DIR}/outputs/pred_test.jsonl"}

# 自动检测可用 GPU 数（也可 NUM_GPUS=4 手动指定）
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-512}
# 每个进程内部固定用 cuda:0（因为已经通过 CUDA_VISIBLE_DEVICES 只暴露了一张卡）
DEVICE_MAP=${DEVICE_MAP:-"cuda:0"}

LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/infer_logs"}
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${PRED_JSONL}")"

# ---- 清理开关 ----
# CLEAN_SHARDS=1（默认）：合并成功后自动删掉 pred_test.rank*.jsonl 分片文件。
#                 =0：保留分片，方便自查 / 断点续跑。
# CLEAN_LOGS=0（默认）：保留 infer_logs/infer_rank*.log，跑挂时可查。
#                 =1：合并成功后连日志一起删。
CLEAN_SHARDS=${CLEAN_SHARDS:-1}
CLEAN_LOGS=${CLEAN_LOGS:-0}

echo "[run_infer_multi_gpu] REPO_ROOT   = ${REPO_ROOT}"
echo "[run_infer_multi_gpu] MODEL_CKPT  = ${MODEL_CKPT}"
echo "[run_infer_multi_gpu] DATA_TEST   = ${DATA_TEST}"
echo "[run_infer_multi_gpu] PRED_JSONL  = ${PRED_JSONL}"
echo "[run_infer_multi_gpu] NUM_GPUS    = ${NUM_GPUS}"
echo "[run_infer_multi_gpu] BATCH_SIZE  = ${BATCH_SIZE}"
echo "[run_infer_multi_gpu] MAX_TOKENS  = ${MAX_TOKENS}"
echo "[run_infer_multi_gpu] DEVICE_MAP  = ${DEVICE_MAP}"
echo "[run_infer_multi_gpu] LOG_DIR     = ${LOG_DIR}"
echo "[run_infer_multi_gpu] CLEAN_SHARDS= ${CLEAN_SHARDS}   # 1=合并后删分片, 0=保留"
echo "[run_infer_multi_gpu] CLEAN_LOGS  = ${CLEAN_LOGS}   # 1=合并后删日志, 0=保留"

# 让子进程能 import 到本仓库里的 qwen_asr（若已 pip install 则冗余但无害）
export PYTHONPATH="${REPO_ROOT}/Qwen3-ASR:${PYTHONPATH:-}"

# 拆分输出文件名（base + ext）——与 infer_test.py 内部生成的分片名保持一致
OUTPUT_BASE="${PRED_JSONL%.*}"
OUTPUT_EXT="${PRED_JSONL##*.}"
if [[ "${OUTPUT_BASE}" == "${PRED_JSONL}" ]]; then
    OUTPUT_EXT=""
fi

# ---- 1) 拉起 N 个子进程 ----
PIDS=()
for ((rank=0; rank<NUM_GPUS; rank++)); do
    LOG="${LOG_DIR}/infer_rank${rank}.log"
    echo "[run_infer_multi_gpu] 启动 rank=${rank} -> GPU ${rank}, 日志: ${LOG}"
    CUDA_VISIBLE_DEVICES=${rank} \
    python "${SCRIPT_DIR}/infer_test.py" \
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

# ---- 2) 等待全部完成（任一失败则整体失败）----
fail=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "[run_infer_multi_gpu][ERROR] 进程 ${pid} 失败"
        fail=1
    fi
done
if [[ ${fail} -ne 0 ]]; then
    echo "[run_infer_multi_gpu][ERROR] 有进程失败，请查看 ${LOG_DIR}/ 下的日志"
    exit 1
fi

# ---- 3) 合并分片 ----
echo "[run_infer_multi_gpu] 合并分片到 ${PRED_JSONL}"
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
        echo "[run_infer_multi_gpu][WARN] 找不到分片: ${SHARD}"
    fi
done

TOTAL=$(wc -l < "${PRED_JSONL}")
echo "[run_infer_multi_gpu][DONE] 合并完成: ${PRED_JSONL}  共 ${TOTAL} 行"

# ---- 4) 可选：清理中间产物（分片 / 日志）----
# 仅在合并成功、且行数 > 0 时才清理，避免把有用产物误删。
if [[ "${TOTAL}" -gt 0 ]]; then
    if [[ "${CLEAN_SHARDS}" == "1" ]]; then
        echo "[run_infer_multi_gpu] CLEAN_SHARDS=1，删除分片文件..."
        for ((rank=0; rank<NUM_GPUS; rank++)); do
            if [[ -n "${OUTPUT_EXT}" ]]; then
                SHARD="${OUTPUT_BASE}.rank${rank}.${OUTPUT_EXT}"
            else
                SHARD="${OUTPUT_BASE}.rank${rank}"
            fi
            [[ -f "${SHARD}" ]] && rm -f "${SHARD}"
        done
    else
        echo "[run_infer_multi_gpu] CLEAN_SHARDS=0，保留分片文件（如需手动清理，见提示）"
        if [[ -n "${OUTPUT_EXT}" ]]; then
            echo "    rm -f ${OUTPUT_BASE}.rank*.${OUTPUT_EXT}"
        else
            echo "    rm -f ${OUTPUT_BASE}.rank*"
        fi
    fi

    if [[ "${CLEAN_LOGS}" == "1" ]]; then
        echo "[run_infer_multi_gpu] CLEAN_LOGS=1，删除进程日志 ${LOG_DIR}/infer_rank*.log"
        rm -f "${LOG_DIR}"/infer_rank*.log
    else
        echo "[run_infer_multi_gpu] CLEAN_LOGS=0，保留进程日志: ${LOG_DIR}/"
    fi
else
    echo "[run_infer_multi_gpu][WARN] 合并结果行数为 0，跳过清理以便排查"
fi

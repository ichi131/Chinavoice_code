#!/usr/bin/env bash
# =============================================================================
# run_hybrid_specialist_infer.sh
# -----------------------------------------------------------------------------
# 基于 LID 置信度的"路由 + 专家改写"混合推理一键脚本：
#   1) 读取 outputs_vc_v2/pred_test_conf.jsonl，按方言阈值路由；
#      对命中 wuyu/kejia/nanchang 高置信度的样本，用对应专用模型改写 pred_text。
#   2) 用 ChinaVoices-Challenge 的 eval 工具计算整体/按方言 CER。
#   3) 用 compute_lid_precision.py 计算 LID 精度。
#   4) 生成 baseline (VC v2) vs hybrid 对比 summary.txt。
#
# 完全独立：所有输出写入 outputs_hybrid_specialist/，不修改任何已有产物。
#
# 环境变量：
#   TAU_PROFILE     阈值 profile：p95 (默认) / p99
#   ROUTE_CONFIG    可选：YAML/JSON 覆写路由表
#   BATCH_SIZE      默认 32
#   MAX_TOKENS      默认 512
#   DEVICE_MAP      默认 cuda:0（每个专用模型独占一张卡）
#   BASE_PRED       默认 outputs_vc_v2/pred_test_conf.jsonl
#   BASELINE_EVAL   默认 ${OUTPUT_ROOT}/baseline_from_conf
#                   —— 与 BASE_PRED 严格对齐的 baseline 评测目录；若不存在会自动生成，
#                   避免与旧的 pred_test.jsonl 评测混用而产生虚假 Δ。
#   OUTPUT_ROOT     默认 outputs_hybrid_specialist
#   OVERWRITE       1 = 覆写已有产物；否则若目标已存在则跳过对应步骤（断点续跑）
#   EVAL_TOOL_SH    官方 CER 评测入口 sh
#   DRY_RUN         1 = 只跑路由决策，不加载专用模型（用于快速核对）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---- 参数 ----
TAU_PROFILE=${TAU_PROFILE:-"p95"}
ROUTE_CONFIG=${ROUTE_CONFIG:-""}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-512}
DEVICE_MAP=${DEVICE_MAP:-"cuda:0"}
BASE_PRED=${BASE_PRED:-"${SCRIPT_DIR}/outputs_vc_v2/pred_test_conf.jsonl"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRIPT_DIR}/outputs_hybrid_specialist"}
# 让 baseline 评测与 BASE_PRED 严格对齐：默认放在 OUTPUT_ROOT 下的独立目录
# （旧路径 outputs_vc_v2/wer_eval 是基于 pred_test.jsonl 的历史产物，会带来 24 条边界样本的虚假 Δ）
BASELINE_EVAL=${BASELINE_EVAL:-"${OUTPUT_ROOT}/baseline_from_conf"}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}
EVAL_TOOL_SH=${EVAL_TOOL_SH:-"/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/ChinaVoices-Challenge/eval/eval_jsonl_with_wer_tools.sh"}

# 根据 profile 分子目录，方便一次跑多种阈值对比
RUN_TAG=${RUN_TAG:-"${TAU_PROFILE}"}
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
PRED_JSONL="${RUN_DIR}/pred_test_routed.jsonl"
WER_DIR="${RUN_DIR}/wer_eval"
LID_TXT="${WER_DIR}/lid_precision.txt"
SUMMARY_TXT="${RUN_DIR}/summary.txt"

mkdir -p "${RUN_DIR}" "${WER_DIR}"

echo "============================================================"
echo "[hybrid] TAU_PROFILE   = ${TAU_PROFILE}"
echo "[hybrid] ROUTE_CONFIG  = ${ROUTE_CONFIG:-'(none)'}"
echo "[hybrid] BATCH_SIZE    = ${BATCH_SIZE}"
echo "[hybrid] MAX_TOKENS    = ${MAX_TOKENS}"
echo "[hybrid] DEVICE_MAP    = ${DEVICE_MAP}"
echo "[hybrid] BASE_PRED     = ${BASE_PRED}"
echo "[hybrid] BASELINE_EVAL = ${BASELINE_EVAL}"
echo "[hybrid] OUTPUT_ROOT   = ${OUTPUT_ROOT}"
echo "[hybrid] RUN_DIR       = ${RUN_DIR}"
echo "[hybrid] OVERWRITE     = ${OVERWRITE}"
echo "[hybrid] DRY_RUN       = ${DRY_RUN}"
echo "[hybrid] EVAL_TOOL_SH  = ${EVAL_TOOL_SH}"
echo "============================================================"

if [[ ! -f "${BASE_PRED}" ]]; then
    echo "[hybrid] ERROR: BASE_PRED not found: ${BASE_PRED}" >&2
    exit 1
fi

if [[ ! -f "${EVAL_TOOL_SH}" ]]; then
    echo "[hybrid] ERROR: EVAL_TOOL_SH not found: ${EVAL_TOOL_SH}" >&2
    exit 1
fi

# =============================================================================
# Step 0: 用 BASE_PRED 自身建立"严格对齐"的 baseline 评测
# -----------------------------------------------------------------------------
# 目的：hybrid 是基于 BASE_PRED（pred_test_conf.jsonl）路由改写而来，若拿
# 旧的 pred_test.jsonl 评测产物做对比，会因为两次独立推理的 GPU bf16 边界
# 抖动（约 24/4934≈0.5% 样本 pred_dialect 翻转）而产生虚假 Δ。
# 这里直接对 BASE_PRED 再跑一遍 CER+LID，产出与 hybrid 完全同一起点的 baseline。
# =============================================================================
mkdir -p "${BASELINE_EVAL}"
BASELINE_LID_TXT="${BASELINE_EVAL}/lid_precision.txt"
if [[ -f "${BASELINE_EVAL}/result.wer" \
      && -f "${BASELINE_EVAL}/by_dialect_summary.txt" \
      && -f "${BASELINE_EVAL}/dialect_accuracy.txt" \
      && -f "${BASELINE_LID_TXT}" \
      && "${OVERWRITE}" != "1" ]]; then
    echo "[hybrid][0/4] SKIP baseline eval: ${BASELINE_EVAL} already complete"
else
    echo "[hybrid][0/4] build aligned baseline eval from BASE_PRED -> ${BASELINE_EVAL}"
    bash "${EVAL_TOOL_SH}" \
        --pred_jsonl "${BASE_PRED}" \
        --output_dir "${BASELINE_EVAL}" \
        --apply_t2s  1 \
        --by_dialect 1
    python "${SCRIPT_DIR}/compute_lid_precision.py" \
        --pred_jsonl "${BASE_PRED}" \
        --out        "${BASELINE_LID_TXT}"
fi

# =============================================================================
# Step 1: 路由 + 专家改写
# =============================================================================
if [[ -f "${PRED_JSONL}" && "${OVERWRITE}" != "1" ]]; then
    echo "[hybrid][1/4] SKIP route: ${PRED_JSONL} already exists (set OVERWRITE=1 to redo)"
else
    echo "[hybrid][1/4] route + specialist rewrite -> ${PRED_JSONL}"
    ROUTE_ARGS=(
        --base_pred_jsonl "${BASE_PRED}"
        --output          "${PRED_JSONL}"
        --tau_profile     "${TAU_PROFILE}"
        --batch-size      "${BATCH_SIZE}"
        --max-tokens      "${MAX_TOKENS}"
        --device-map      "${DEVICE_MAP}"
        --dry_run         "${DRY_RUN}"
    )
    if [[ -n "${ROUTE_CONFIG}" ]]; then
        ROUTE_ARGS+=( --route_config "${ROUTE_CONFIG}" )
    fi
    python "${SCRIPT_DIR}/route_by_confidence.py" "${ROUTE_ARGS[@]}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[hybrid] DRY_RUN=1，跳过 CER / LID / summary。"
    exit 0
fi

if [[ ! -f "${PRED_JSONL}" ]]; then
    echo "[hybrid] ERROR: route step did not produce ${PRED_JSONL}" >&2
    exit 1
fi

# =============================================================================
# Step 2: CER 评测（整体 + 按方言）
# =============================================================================
if [[ -f "${WER_DIR}/result.wer" && -f "${WER_DIR}/by_dialect_summary.txt" && "${OVERWRITE}" != "1" ]]; then
    echo "[hybrid][2/4] SKIP CER: ${WER_DIR}/result.wer already exists"
else
    echo "[hybrid][2/4] compute CER via ${EVAL_TOOL_SH}"
    bash "${EVAL_TOOL_SH}" \
        --pred_jsonl "${PRED_JSONL}" \
        --output_dir "${WER_DIR}" \
        --apply_t2s  1 \
        --by_dialect 1
fi

# =============================================================================
# Step 3: LID 精度
# =============================================================================
if [[ -f "${LID_TXT}" && "${OVERWRITE}" != "1" ]]; then
    echo "[hybrid][3/4] SKIP LID: ${LID_TXT} already exists"
else
    echo "[hybrid][3/4] compute LID precision -> ${LID_TXT}"
    python "${SCRIPT_DIR}/compute_lid_precision.py" \
        --pred_jsonl "${PRED_JSONL}" \
        --out        "${LID_TXT}"
fi

# =============================================================================
# Step 4: baseline vs hybrid 对比 summary
# =============================================================================
if [[ -f "${SUMMARY_TXT}" && "${OVERWRITE}" != "1" ]]; then
    echo "[hybrid][4/4] SKIP summary: ${SUMMARY_TXT} already exists"
else
    echo "[hybrid][4/4] build summary -> ${SUMMARY_TXT}"
    python "${SCRIPT_DIR}/route_summarize.py" \
        --baseline_eval_dir "${BASELINE_EVAL}" \
        --hybrid_eval_dir   "${WER_DIR}" \
        --routed_jsonl      "${PRED_JSONL}" \
        --run_tag           "${RUN_TAG}" \
        --tau_profile       "${TAU_PROFILE}" \
        --output            "${SUMMARY_TXT}"
fi

echo "============================================================"
echo "[hybrid] done."
echo "  routed jsonl : ${PRED_JSONL}"
echo "  CER overall  : ${WER_DIR}/result.wer"
echo "  CER by dial. : ${WER_DIR}/by_dialect_summary.txt"
echo "  LID precision: ${LID_TXT}"
echo "  summary      : ${SUMMARY_TXT}"
echo "============================================================"
cat "${SUMMARY_TXT}" || true

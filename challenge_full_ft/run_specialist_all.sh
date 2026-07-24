#!/usr/bin/env bash
# =============================================================================
# run_specialist_all.sh
# -----------------------------------------------------------------------------
# 一键为 16 种方言各训一个单语种专用 ASR 模型（串行），并汇总对比结果。
#
# 特殊处理：
#   - wuyu 已在 outputs_wuyu_stage2 里训好，默认跳过再训练；直接复用其
#     wer_eval/by_dialect/wuyu/result.wer（或 outputs_wuyu_stage2/wer_eval/result.wer）
#     纳入最终对比表。可用 RUN_WUYU=1 强制重训。
#
# 流程：
#   for accent in 16 方言:
#       ACCENT=$accent bash run_specialist_pipeline.sh
#   汇总所有 wer_eval/result.wer 与 VC v2 联合模型 baseline 到
#       outputs_specialist_compare/summary.txt
#
# 用法：
#   bash challenge_full_ft/run_specialist_all.sh 2>&1 | tee run_specialist_all.log
#
# 常用环境变量：
#   DIALECTS           空格分隔的方言子集（默认全跑 16 个）
#   RUN_WUYU           1/0(默认)  是否重训 wuyu（默认复用已有结果）
#   FORCE_REBUILD      1/0(默认)  强制重建数据
#   FORCE_TRAIN        1/0(默认)  强制重训
#   FORCE_EVAL         1/0(默认)  强制重推理
#   CONTINUE_ON_ERROR  1(默认)/0  某方言失败是否继续下一个
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- 方言列表 ----------
ALL_ACCENTS=(anhui cantonese changsha chaoshan dongbei henan kejia minnan
             nanchang nanjing shan1xi shan3xi shandong sichuan wuhan wuyu)

DIALECTS=${DIALECTS:-"${ALL_ACCENTS[*]}"}
read -ra RUN_LIST <<< "${DIALECTS}"

# ---------- 控制变量 ----------
RUN_WUYU=${RUN_WUYU:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
FORCE_REBUILD=${FORCE_REBUILD:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_EVAL=${FORCE_EVAL:-0}

# ---------- 路径 ----------
DATA_ROOT=${DATA_ROOT:-"${SCRIPT_DIR}/data_specialist"}
OUT_ROOT=${OUT_ROOT:-"${SCRIPT_DIR}/outputs_specialist"}
COMPARE_OUT=${COMPARE_OUT:-"${SCRIPT_DIR}/outputs_specialist_compare"}
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs_specialist"}
VC_V2_BY_DIALECT=${VC_V2_BY_DIALECT:-"${SCRIPT_DIR}/outputs_vc_v2/wer_eval/by_dialect_summary.txt"}

# wuyu 已有结果的目录（stage2 是最优的 2-stage 模型）
WUYU_EXISTING_OUT=${WUYU_EXISTING_OUT:-"${SCRIPT_DIR}/outputs_wuyu_stage2"}

mkdir -p "${OUT_ROOT}" "${COMPARE_OUT}" "${LOG_DIR}"

echo "============================================================"
echo "[run_specialist_all] DIALECTS         = ${RUN_LIST[*]}"
echo "[run_specialist_all] RUN_WUYU         = ${RUN_WUYU}"
echo "[run_specialist_all] CONTINUE_ON_ERROR= ${CONTINUE_ON_ERROR}"
echo "[run_specialist_all] FORCE_REBUILD    = ${FORCE_REBUILD}"
echo "[run_specialist_all] FORCE_TRAIN      = ${FORCE_TRAIN}"
echo "[run_specialist_all] FORCE_EVAL       = ${FORCE_EVAL}"
echo "[run_specialist_all] DATA_ROOT        = ${DATA_ROOT}"
echo "[run_specialist_all] OUT_ROOT         = ${OUT_ROOT}"
echo "[run_specialist_all] COMPARE_OUT      = ${COMPARE_OUT}"
echo "[run_specialist_all] LOG_DIR          = ${LOG_DIR}"
echo "[run_specialist_all] VC_V2_BY_DIALECT = ${VC_V2_BY_DIALECT}"
echo "[run_specialist_all] WUYU_EXISTING    = ${WUYU_EXISTING_OUT}"
echo "============================================================"

# =============================================================================
# 主循环
# =============================================================================
declare -A STATUS_MAP    # accent -> ok/failed/skipped/reused
FAILED=()

for ACCENT in "${RUN_LIST[@]}"; do
    echo
    echo "============================================================"
    echo "[run_specialist_all] >>> 处理方言: ${ACCENT}"
    echo "============================================================"

    # wuyu 特殊处理：默认复用已有结果
    if [[ "${ACCENT}" == "wuyu" && "${RUN_WUYU}" != "1" ]]; then
        echo "[run_specialist_all:wuyu] 复用已有结果：${WUYU_EXISTING_OUT}"
        if [[ ! -s "${WUYU_EXISTING_OUT}/wer_eval/result.wer" ]]; then
            echo "[run_specialist_all:wuyu] WARN: 未找到 ${WUYU_EXISTING_OUT}/wer_eval/result.wer"
            STATUS_MAP[${ACCENT}]="reused_missing"
        else
            STATUS_MAP[${ACCENT}]="reused"
        fi
        continue
    fi

    log_file="${LOG_DIR}/${ACCENT}.log"
    echo "[run_specialist_all:${ACCENT}] 日志: ${log_file}"

    set +e
    ACCENT="${ACCENT}" \
    FORCE_REBUILD="${FORCE_REBUILD}" \
    FORCE_TRAIN="${FORCE_TRAIN}" \
    FORCE_EVAL="${FORCE_EVAL}" \
    bash "${SCRIPT_DIR}/run_specialist_pipeline.sh" "${ACCENT}" \
        2>&1 | tee "${log_file}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "${rc}" -eq 0 ]]; then
        echo "[run_specialist_all:${ACCENT}] OK"
        STATUS_MAP[${ACCENT}]="ok"
    else
        echo "[run_specialist_all:${ACCENT}] FAILED (rc=${rc})"
        STATUS_MAP[${ACCENT}]="failed"
        FAILED+=("${ACCENT}")
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
            echo "[run_specialist_all] CONTINUE_ON_ERROR=0，终止" >&2
            break
        fi
    fi
done

# =============================================================================
# 汇总
# =============================================================================
echo
echo "============================================================"
echo "[run_specialist_all] 生成汇总表 -> ${COMPARE_OUT}/summary.txt"
echo "============================================================"

# 解析 result.wer 的 python 一行
parse_cer_py=$(cat <<'PYEOF'
import re, sys
try:
    with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
except Exception:
    print("N/A"); sys.exit(0)
pats = [
    r"Overall[^\n]*?([0-9]+\.[0-9]+)\s*%",
    r"%WER\s+([0-9]+\.[0-9]+)",
    r"WER\s*=\s*([0-9]+\.[0-9]+)",
    r"CER\s*[:=]\s*([0-9]+\.[0-9]+)",
    r"([0-9]+\.[0-9]+)\s*%",
]
for p in pats:
    m = re.search(p, txt)
    if m:
        print(m.group(1)); sys.exit(0)
print("N/A")
PYEOF
)

# 解析 VC v2 by_dialect_summary.txt 得到 baseline，格式：
#   dialect                 samples          wer
#   ----------------------------------------------
#   anhui                       350       14.00%
parse_vc_baseline_py=$(cat <<'PYEOF'
import sys, re
path = sys.argv[1]
out = {}
try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("dialect"):
                continue
            parts = re.split(r"\s+", line)
            # 期望三列：dialect samples wer%
            if len(parts) >= 3 and re.match(r"^[0-9]+\.[0-9]+%?$", parts[-1]):
                dia = parts[0]
                wer = parts[-1].rstrip("%")
                out[dia] = wer
except Exception:
    pass
for k, v in out.items():
    print(f"{k}\t{v}")
PYEOF
)

# 读入 VC v2 baseline
declare -A VC_BASELINE
if [[ -s "${VC_V2_BY_DIALECT}" ]]; then
    while IFS=$'\t' read -r dia wer; do
        [[ -n "${dia}" ]] && VC_BASELINE[${dia}]="${wer}"
    done < <(python -c "${parse_vc_baseline_py}" "${VC_V2_BY_DIALECT}")
    echo "[run_specialist_all] 已加载 VC v2 baseline（${#VC_BASELINE[@]} 方言）"
else
    echo "[run_specialist_all] WARN: VC v2 by_dialect_summary 不存在: ${VC_V2_BY_DIALECT}"
fi

get_cer() {
    local wer_file="$1"
    if [[ ! -f "${wer_file}" ]]; then echo "N/A"; return; fi
    python -c "${parse_cer_py}" "${wer_file}"
}

# 每个方言取 result.wer
declare -A SPEC_CER SPEC_CKPT SPEC_TEST_N

for ACCENT in "${RUN_LIST[@]}"; do
    if [[ "${ACCENT}" == "wuyu" && "${RUN_WUYU}" != "1" ]]; then
        wer_file="${WUYU_EXISTING_OUT}/wer_eval/result.wer"
        ckpt_file="${WUYU_EXISTING_OUT}/best_ckpt.txt"
        test_file="${SCRIPT_DIR}/data_wuyu_stage2/test_wuyu.jsonl"
    else
        wer_file="${OUT_ROOT}/${ACCENT}/wer_eval/result.wer"
        ckpt_file="${OUT_ROOT}/${ACCENT}/best_ckpt.txt"
        test_file="${DATA_ROOT}/${ACCENT}/test.jsonl"
    fi
    SPEC_CER[${ACCENT}]=$(get_cer "${wer_file}")
    if [[ -s "${ckpt_file}" ]]; then
        SPEC_CKPT[${ACCENT}]="$(cat "${ckpt_file}" | tr -d '[:space:]')"
    else
        SPEC_CKPT[${ACCENT}]="N/A"
    fi
    if [[ -s "${test_file}" ]]; then
        SPEC_TEST_N[${ACCENT}]="$(wc -l < "${test_file}")"
    else
        SPEC_TEST_N[${ACCENT}]="?"
    fi
done

# 计算 Δ = specialist - VC v2 baseline
compute_delta() {
    local a="$1"; local b="$2"
    if [[ "${a}" == "N/A" || "${b}" == "N/A" || -z "${b}" ]]; then
        echo "N/A"; return
    fi
    python -c "import sys; print(f'{float(sys.argv[1])-float(sys.argv[2]):+.2f}')" "${a}" "${b}"
}

SUMMARY_FILE="${COMPARE_OUT}/summary.txt"
{
    echo "======================================================================"
    echo " 16 方言 单语种专用 ASR (specialist) vs VC v2 联合模型 baseline"
    echo "----------------------------------------------------------------------"
    echo " * specialist_cer : 单语种 specialist 模型在该方言 test 上的 CER (%)"
    echo " * baseline_cer   : outputs_vc_v2/wer_eval/by_dialect_summary.txt 中该"
    echo "                    方言的 CER (%)"
    echo " * Δ              : specialist_cer - baseline_cer  (负值表示 specialist 更好)"
    echo " * status         : 训练状态；'reused' 表示复用既有产物（wuyu）"
    echo "======================================================================"
    printf "%-10s  %-8s  %-13s  %-11s  %-8s  %-8s  %s\n" \
        "dialect" "test_n" "specialist(%)" "baseline(%)" "Δ" "status" "best_ckpt"
    printf "%-10s  %-8s  %-13s  %-11s  %-8s  %-8s  %s\n" \
        "----------" "--------" "-------------" "-----------" "--------" "--------" "---------"
    for ACCENT in "${ALL_ACCENTS[@]}"; do
        s_cer="${SPEC_CER[${ACCENT}]:-N/A}"
        b_cer="${VC_BASELINE[${ACCENT}]:-N/A}"
        delta=$(compute_delta "${s_cer}" "${b_cer}")
        status="${STATUS_MAP[${ACCENT}]:-not_run}"
        ckpt="${SPEC_CKPT[${ACCENT}]:-N/A}"
        n="${SPEC_TEST_N[${ACCENT}]:-?}"
        printf "%-10s  %-8s  %-13s  %-11s  %-8s  %-8s  %s\n" \
            "${ACCENT}" "${n}" "${s_cer}" "${b_cer}" "${delta}" "${status}" "${ckpt}"
    done
    echo "======================================================================"
    if [[ "${#FAILED[@]}" -gt 0 ]]; then
        echo " 失败方言: ${FAILED[*]}"
    else
        echo " 无失败方言。"
    fi
    echo "======================================================================"
} | tee "${SUMMARY_FILE}"

# =============================================================================
# 尾声
# =============================================================================
echo
echo "============================================================"
echo "[run_specialist_all] ★ 全部完成"
echo "  Summary : ${SUMMARY_FILE}"
echo "  Logs    : ${LOG_DIR}/*.log"
echo "============================================================"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
    exit 2
fi
exit 0

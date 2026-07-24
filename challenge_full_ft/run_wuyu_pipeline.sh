#!/usr/bin/env bash
# =============================================================================
# run_wuyu_pipeline.sh
# -----------------------------------------------------------------------------
# 吴语（wuyu）专用 ASR 两阶段微调一键 pipeline。
#
# 流程（可断点续跑，中间产物已存在即跳过）：
#   [1] 数据规范化：
#       - 50h SFT 化           -> data_wuyu_50h/all.jsonl
#       - VC_data_v2 wuyu 抽取 -> data_wuyu_stage2/_vc_train.jsonl / _vc_val.jsonl
#         （build_stage2 内部会重新过滤，这里仅落地作缓存不是必需，故直接由
#          build_stage2 子命令即时处理原始文件，无需中间文件）
#   [2] Stage 1 数据切分     -> data_wuyu_50h/{train,val}.jsonl
#   [3] Stage 2 数据合并去重 -> data_wuyu_stage2/{train,val}.jsonl
#   [4] 提取 test wuyu       -> data_wuyu_stage2/test_wuyu.jsonl
#   [5] Stage 1 训练         -> outputs_wuyu_stage1/
#   [6] Stage 2 训练         -> outputs_wuyu_stage2/  （base = Stage 1 best）
#   [7] Control 训练         -> outputs_wuyu_ctrl/    （base = Qwen3-ASR-1.7B）
#                              默认开启，可用 RUN_CTRL=0 关闭
#   [8] 三方评估             -> outputs_wuyu_*/wer_eval/
#   [9] 三方对比汇总         -> outputs_wuyu_compare/summary.txt
#
# 用法：
#   bash challenge_full_ft/run_wuyu_pipeline.sh
# 或：
#   bash challenge_full_ft/run_wuyu_pipeline.sh 2>&1 | tee run_wuyu_pipeline.log
#
# 常用环境变量：
#   RUN_CTRL           1(默认)/0     是否训练+评估对照组
#   FORCE_REBUILD      1/0(默认)     强制重建所有数据产物
#   FORCE_STAGE1       1/0(默认)     强制重训 Stage 1
#   FORCE_STAGE2       1/0(默认)     强制重训 Stage 2
#   FORCE_CTRL         1/0(默认)     强制重训 Control
#   FORCE_EVAL         1/0(默认)     强制重推理（对已有 pred_wuyu_test.jsonl 覆盖）
# =============================================================================

set -euo pipefail

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

# 原始数据源
WENETSPEECH_50H_JSONL=${WENETSPEECH_50H_JSONL:-"/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/challenge-data/wenetspeech-wu/audios_50h.jsonl"}
VC_V2_DIR=${VC_V2_DIR:-"${ROOT_DIR}/VC_data_v2"}
VC_V2_TRAIN=${VC_V2_TRAIN:-"${VC_V2_DIR}/data_train_vc.jsonl"}
VC_V2_VAL=${VC_V2_VAL:-"${VC_V2_DIR}/data_val_vc.jsonl"}
RAW_TRAIN=${RAW_TRAIN:-"${SCRIPT_DIR}/data/train.jsonl"}
RAW_VAL=${RAW_VAL:-"${SCRIPT_DIR}/data/val.jsonl"}
RAW_TEST=${RAW_TEST:-"${SCRIPT_DIR}/data/test.jsonl"}

# 产物目录
STAGE1_DATA_DIR=${STAGE1_DATA_DIR:-"${SCRIPT_DIR}/data_wuyu_50h"}
STAGE2_DATA_DIR=${STAGE2_DATA_DIR:-"${SCRIPT_DIR}/data_wuyu_stage2"}
STAGE1_OUT=${STAGE1_OUT:-"${SCRIPT_DIR}/outputs_wuyu_stage1"}
STAGE2_OUT=${STAGE2_OUT:-"${SCRIPT_DIR}/outputs_wuyu_stage2"}
CTRL_OUT=${CTRL_OUT:-"${SCRIPT_DIR}/outputs_wuyu_ctrl"}
COMPARE_OUT=${COMPARE_OUT:-"${SCRIPT_DIR}/outputs_wuyu_compare"}

BASE_MODEL=${BASE_MODEL:-"/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B"}

# ---------- 训练超参（严格对齐 baseline / VC v2 系列） ----------
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-5}
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
RUN_CTRL=${RUN_CTRL:-1}
FORCE_REBUILD=${FORCE_REBUILD:-0}
FORCE_STAGE1=${FORCE_STAGE1:-0}
FORCE_STAGE2=${FORCE_STAGE2:-0}
FORCE_CTRL=${FORCE_CTRL:-0}
FORCE_EVAL=${FORCE_EVAL:-0}

# 50h 规范化中间产物
NORM_50H_JSONL="${STAGE1_DATA_DIR}/all.jsonl"

# ---------- 打印配置 ----------
echo "============================================================"
echo "[run_wuyu_pipeline] === 配置摘要 ==="
echo "[run_wuyu_pipeline] WENETSPEECH_50H_JSONL = ${WENETSPEECH_50H_JSONL}"
echo "[run_wuyu_pipeline] VC_V2_TRAIN           = ${VC_V2_TRAIN}"
echo "[run_wuyu_pipeline] VC_V2_VAL             = ${VC_V2_VAL}"
echo "[run_wuyu_pipeline] RAW_TRAIN             = ${RAW_TRAIN}"
echo "[run_wuyu_pipeline] RAW_VAL               = ${RAW_VAL}"
echo "[run_wuyu_pipeline] RAW_TEST              = ${RAW_TEST}"
echo "------------------------------------------------------------"
echo "[run_wuyu_pipeline] STAGE1_DATA_DIR       = ${STAGE1_DATA_DIR}"
echo "[run_wuyu_pipeline] STAGE2_DATA_DIR       = ${STAGE2_DATA_DIR}"
echo "[run_wuyu_pipeline] STAGE1_OUT            = ${STAGE1_OUT}"
echo "[run_wuyu_pipeline] STAGE2_OUT            = ${STAGE2_OUT}"
echo "[run_wuyu_pipeline] CTRL_OUT              = ${CTRL_OUT}"
echo "[run_wuyu_pipeline] COMPARE_OUT           = ${COMPARE_OUT}"
echo "[run_wuyu_pipeline] BASE_MODEL            = ${BASE_MODEL}"
echo "------------------------------------------------------------"
echo "[run_wuyu_pipeline] NUM_GPUS              = ${NUM_GPUS}"
echo "[run_wuyu_pipeline] EPOCHS                = ${EPOCHS}"
echo "[run_wuyu_pipeline] BATCH_SIZE            = ${BATCH_SIZE}"
echo "[run_wuyu_pipeline] GRAD_ACC              = ${GRAD_ACC}"
echo "[run_wuyu_pipeline] LR                    = ${LR}"
echo "[run_wuyu_pipeline] SAVE_STEPS            = ${SAVE_STEPS}"
echo "[run_wuyu_pipeline] SAVE_TOTAL_LIMIT      = ${SAVE_TOTAL_LIMIT}"
echo "[run_wuyu_pipeline] EARLY_STOP_PATIENCE   = ${EARLY_STOP_PATIENCE}"
echo "[run_wuyu_pipeline] EARLY_STOP_THRESHOLD  = ${EARLY_STOP_THRESHOLD}"
echo "[run_wuyu_pipeline] BF16                  = ${BF16_VAL}"
echo "------------------------------------------------------------"
echo "[run_wuyu_pipeline] RUN_CTRL              = ${RUN_CTRL}"
echo "[run_wuyu_pipeline] FORCE_REBUILD         = ${FORCE_REBUILD}"
echo "[run_wuyu_pipeline] FORCE_STAGE1          = ${FORCE_STAGE1}"
echo "[run_wuyu_pipeline] FORCE_STAGE2          = ${FORCE_STAGE2}"
echo "[run_wuyu_pipeline] FORCE_CTRL            = ${FORCE_CTRL}"
echo "[run_wuyu_pipeline] FORCE_EVAL            = ${FORCE_EVAL}"
echo "============================================================"

# ---------- 前置检查 ----------
for f in "${WENETSPEECH_50H_JSONL}" "${VC_V2_TRAIN}" "${VC_V2_VAL}" \
         "${RAW_TRAIN}" "${RAW_VAL}" "${RAW_TEST}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[run_wuyu_pipeline] ERROR: 输入文件不存在: ${f}" >&2
        exit 1
    fi
done

mkdir -p "${STAGE1_DATA_DIR}" "${STAGE2_DATA_DIR}"

FORCE_FLAG=""
if [[ "${FORCE_REBUILD}" == "1" ]]; then
    FORCE_FLAG="--force"
fi

# =============================================================================
# [Step 1] 数据规范化 —— 50h
# =============================================================================
echo
echo "[run_wuyu_pipeline][Step 1/9] 规范化 50h -> ${NORM_50H_JSONL}"
python "${SCRIPT_DIR}/prepare_wuyu_data.py" normalize \
    --source wenetspeech50h \
    --src "${WENETSPEECH_50H_JSONL}" \
    --dst "${NORM_50H_JSONL}" \
    --check_audio_exists 0 \
    ${FORCE_FLAG}

# =============================================================================
# [Step 2] Stage 1 数据切分
# =============================================================================
echo
echo "[run_wuyu_pipeline][Step 2/9] Stage 1 切分 5% val -> ${STAGE1_DATA_DIR}"
python "${SCRIPT_DIR}/prepare_wuyu_data.py" split_50h \
    --src "${NORM_50H_JSONL}" \
    --out_dir "${STAGE1_DATA_DIR}" \
    --val_ratio 0.05 \
    ${FORCE_FLAG}

STAGE1_TRAIN="${STAGE1_DATA_DIR}/train.jsonl"
STAGE1_VAL="${STAGE1_DATA_DIR}/val.jsonl"
if [[ ! -s "${STAGE1_TRAIN}" || ! -s "${STAGE1_VAL}" ]]; then
    echo "[run_wuyu_pipeline] ERROR: Stage 1 数据切分失败" >&2
    exit 1
fi

# =============================================================================
# [Step 3] Stage 2 数据合并去重
# =============================================================================
echo
echo "[run_wuyu_pipeline][Step 3/9] Stage 2 合并去重 -> ${STAGE2_DATA_DIR}"
python "${SCRIPT_DIR}/prepare_wuyu_data.py" build_stage2 \
    --vc_train "${VC_V2_TRAIN}" \
    --vc_val "${VC_V2_VAL}" \
    --raw_train "${RAW_TRAIN}" \
    --raw_val "${RAW_VAL}" \
    --out_dir "${STAGE2_DATA_DIR}" \
    --check_audio_exists 0 \
    ${FORCE_FLAG}

STAGE2_TRAIN="${STAGE2_DATA_DIR}/train.jsonl"
STAGE2_VAL="${STAGE2_DATA_DIR}/val.jsonl"
if [[ ! -s "${STAGE2_TRAIN}" || ! -s "${STAGE2_VAL}" ]]; then
    echo "[run_wuyu_pipeline] ERROR: Stage 2 数据合并失败" >&2
    exit 1
fi

# =============================================================================
# [Step 4] 提取 test wuyu
# =============================================================================
TEST_WUYU="${STAGE2_DATA_DIR}/test_wuyu.jsonl"
echo
echo "[run_wuyu_pipeline][Step 4/9] 提取 test wuyu -> ${TEST_WUYU}"
python "${SCRIPT_DIR}/prepare_wuyu_data.py" extract_test_wuyu \
    --src "${RAW_TEST}" \
    --dst "${TEST_WUYU}"

echo
echo "[run_wuyu_pipeline] --- 数据准备阶段完成 ---"
echo "[run_wuyu_pipeline]     Stage 1 train  = $(wc -l < "${STAGE1_TRAIN}")"
echo "[run_wuyu_pipeline]     Stage 1 val    = $(wc -l < "${STAGE1_VAL}")"
echo "[run_wuyu_pipeline]     Stage 2 train  = $(wc -l < "${STAGE2_TRAIN}")"
echo "[run_wuyu_pipeline]     Stage 2 val    = $(wc -l < "${STAGE2_VAL}")"
echo "[run_wuyu_pipeline]     test wuyu      = $(wc -l < "${TEST_WUYU}")"

# =============================================================================
# 工具函数：判断 output_dir 中的 best_ckpt.txt 是否有效
# =============================================================================
has_valid_best_ckpt() {
    local out_dir="$1"
    local best_file="${out_dir}/best_ckpt.txt"
    if [[ ! -s "${best_file}" ]]; then
        return 1
    fi
    local ckpt
    ckpt="$(cat "${best_file}" | tr -d '[:space:]')"
    if [[ -z "${ckpt}" || ! -f "${ckpt}/config.json" ]]; then
        return 1
    fi
    return 0
}

# 通用的训练调用函数
run_train() {
    local model_path="$1"
    local train_file="$2"
    local eval_file="$3"
    local out_dir="$4"
    local tag="$5"

    mkdir -p "${out_dir}"
    echo "[run_wuyu_pipeline:${tag}] START train"
    echo "[run_wuyu_pipeline:${tag}]   MODEL_PATH = ${model_path}"
    echo "[run_wuyu_pipeline:${tag}]   TRAIN_FILE = ${train_file}"
    echo "[run_wuyu_pipeline:${tag}]   EVAL_FILE  = ${eval_file}"
    echo "[run_wuyu_pipeline:${tag}]   OUTPUT_DIR = ${out_dir}"

    NUM_GPUS="${NUM_GPUS}" \
    MODEL_PATH="${model_path}" \
    TRAIN_FILE="${train_file}" \
    EVAL_FILE="${eval_file}" \
    OUTPUT_DIR="${out_dir}" \
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

    if ! has_valid_best_ckpt "${out_dir}"; then
        echo "[run_wuyu_pipeline:${tag}] ERROR: 训练结束但 best_ckpt.txt 无效" >&2
        exit 1
    fi
    echo "[run_wuyu_pipeline:${tag}] DONE. best = $(cat "${out_dir}/best_ckpt.txt")"
}

# =============================================================================
# [Step 5] Stage 1 训练
# =============================================================================
echo
echo "[run_wuyu_pipeline][Step 5/9] Stage 1 训练"
if has_valid_best_ckpt "${STAGE1_OUT}" && [[ "${FORCE_STAGE1}" != "1" ]]; then
    echo "[run_wuyu_pipeline:stage1] 已存在有效 best_ckpt，SKIP"
    echo "[run_wuyu_pipeline:stage1]   best = $(cat "${STAGE1_OUT}/best_ckpt.txt")"
else
    run_train "${BASE_MODEL}" "${STAGE1_TRAIN}" "${STAGE1_VAL}" \
              "${STAGE1_OUT}" "stage1"
fi

STAGE1_BEST="$(cat "${STAGE1_OUT}/best_ckpt.txt")"
if [[ -z "${STAGE1_BEST}" || ! -f "${STAGE1_BEST}/config.json" ]]; then
    echo "[run_wuyu_pipeline] ERROR: Stage 1 best ckpt 无效: ${STAGE1_BEST}" >&2
    exit 1
fi

# =============================================================================
# [Step 6] Stage 2 训练（在 Stage 1 best 基础上）
# =============================================================================
echo
echo "[run_wuyu_pipeline][Step 6/9] Stage 2 训练（base = Stage 1 best）"
if has_valid_best_ckpt "${STAGE2_OUT}" && [[ "${FORCE_STAGE2}" != "1" ]]; then
    echo "[run_wuyu_pipeline:stage2] 已存在有效 best_ckpt，SKIP"
    echo "[run_wuyu_pipeline:stage2]   best = $(cat "${STAGE2_OUT}/best_ckpt.txt")"
else
    run_train "${STAGE1_BEST}" "${STAGE2_TRAIN}" "${STAGE2_VAL}" \
              "${STAGE2_OUT}" "stage2"
fi

# =============================================================================
# [Step 7] Control 训练（在 base 上，仅用 Stage 2 数据）
# =============================================================================
echo
echo "[run_wuyu_pipeline][Step 7/9] Control 训练（base = Qwen3-ASR-1.7B）"
if [[ "${RUN_CTRL}" != "1" ]]; then
    echo "[run_wuyu_pipeline:ctrl] RUN_CTRL=${RUN_CTRL}，跳过对照组"
elif has_valid_best_ckpt "${CTRL_OUT}" && [[ "${FORCE_CTRL}" != "1" ]]; then
    echo "[run_wuyu_pipeline:ctrl] 已存在有效 best_ckpt，SKIP"
    echo "[run_wuyu_pipeline:ctrl]   best = $(cat "${CTRL_OUT}/best_ckpt.txt")"
else
    run_train "${BASE_MODEL}" "${STAGE2_TRAIN}" "${STAGE2_VAL}" \
              "${CTRL_OUT}" "ctrl"
fi

# =============================================================================
# [Step 8] 三方评估
# =============================================================================
echo
echo "[run_wuyu_pipeline][Step 8/9] 三方评估"

# 单模型评估函数：容错，单个失败不中断其它
eval_one() {
    local tag="$1"
    local out_dir="$2"
    local pred_jsonl="${out_dir}/pred_wuyu_test.jsonl"
    local wer_dir="${out_dir}/wer_eval"

    if ! has_valid_best_ckpt "${out_dir}"; then
        echo "[run_wuyu_pipeline:eval:${tag}] WARN: best_ckpt 无效，跳过"
        return 1
    fi

    # 若已有非空 pred_wuyu_test.jsonl 且未强制重推理，仍可直接重跑 WER 部分
    # （run_eval.sh 会再跑一遍推理；这里的粗略优化：删除已有的 pred 才能触发重推）
    if [[ -s "${pred_jsonl}" && "${FORCE_EVAL}" != "1" ]]; then
        echo "[run_wuyu_pipeline:eval:${tag}] pred 已存在，仅重算 CER。"
        # run_eval.sh 会覆盖 pred_jsonl，为避免重复推理这里绕开推理，
        # 直接调用官方 CER 工具（EVAL_TOOL_SH）
        local eval_tool="/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/ChinaVoices-Challenge/eval/eval_jsonl_with_wer_tools.sh"
        if [[ ! -f "${eval_tool}" ]]; then
            echo "[run_wuyu_pipeline:eval:${tag}] WARN: eval tool 不存在，回退到完整 run_eval.sh"
        else
            mkdir -p "${wer_dir}"
            if bash "${eval_tool}" \
                --pred_jsonl "${pred_jsonl}" \
                --output_dir "${wer_dir}" \
                --apply_t2s  1 \
                --by_dialect 1; then
                echo "[run_wuyu_pipeline:eval:${tag}] DONE (CER-only)"
                return 0
            else
                echo "[run_wuyu_pipeline:eval:${tag}] CER-only 失败，回退到完整 run_eval.sh"
            fi
        fi
    fi

    echo "[run_wuyu_pipeline:eval:${tag}] 完整推理 + CER"
    if OUTPUT_DIR="${out_dir}" \
       DATA_TEST="${TEST_WUYU}" \
       PRED_JSONL="${pred_jsonl}" \
       WER_DIR="${wer_dir}" \
       bash "${SCRIPT_DIR}/run_eval.sh"; then
        echo "[run_wuyu_pipeline:eval:${tag}] DONE"
        return 0
    else
        echo "[run_wuyu_pipeline:eval:${tag}] ERROR: 评估失败"
        return 1
    fi
}

# 用 || true 让单个失败不中断
eval_one "stage1" "${STAGE1_OUT}" || echo "[run_wuyu_pipeline] stage1 eval FAILED, 继续"
eval_one "stage2" "${STAGE2_OUT}" || echo "[run_wuyu_pipeline] stage2 eval FAILED, 继续"
if [[ "${RUN_CTRL}" == "1" ]]; then
    eval_one "ctrl" "${CTRL_OUT}" || echo "[run_wuyu_pipeline] ctrl eval FAILED, 继续"
fi

# =============================================================================
# [Step 9] 对比汇总
# =============================================================================
echo
echo "[run_wuyu_pipeline][Step 9/9] 对比汇总 -> ${COMPARE_OUT}/summary.txt"
mkdir -p "${COMPARE_OUT}"

# 解析 result.wer 中的整体 CER
# 官方 eval 工具的输出格式：文件里含 "Overall %WER = xx.xx" 或首行 %WER 数字
# 这里用 python 稳健解析
parse_cer_py=$(cat <<'PYEOF'
import re, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
except Exception as e:
    print(f"N/A")
    sys.exit(0)
# 优先匹配 "Overall .* WER" / "WER = xx.xx" / "%WER xx.xx"
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
        print(m.group(1))
        sys.exit(0)
print("N/A")
PYEOF
)

get_cer() {
    local wer_file="$1"
    if [[ ! -f "${wer_file}" ]]; then
        echo "N/A"
        return
    fi
    python -c "${parse_cer_py}" "${wer_file}"
}

STAGE1_CER=$(get_cer "${STAGE1_OUT}/wer_eval/result.wer")
STAGE2_CER=$(get_cer "${STAGE2_OUT}/wer_eval/result.wer")
CTRL_CER="N/A"
if [[ "${RUN_CTRL}" == "1" ]]; then
    CTRL_CER=$(get_cer "${CTRL_OUT}/wer_eval/result.wer")
fi

STAGE1_BEST_TXT=$(cat "${STAGE1_OUT}/best_ckpt.txt" 2>/dev/null || echo "N/A")
STAGE2_BEST_TXT=$(cat "${STAGE2_OUT}/best_ckpt.txt" 2>/dev/null || echo "N/A")
CTRL_BEST_TXT=$(cat "${CTRL_OUT}/best_ckpt.txt" 2>/dev/null || echo "N/A")

# 计算 Δ(vs Control)
compute_delta() {
    local this_cer="$1"
    local ref_cer="$2"
    if [[ "${this_cer}" == "N/A" || "${ref_cer}" == "N/A" ]]; then
        echo "N/A"
        return
    fi
    python -c "import sys; a=float(sys.argv[1]); b=float(sys.argv[2]); print(f'{a-b:+.2f}')" "${this_cer}" "${ref_cer}"
}

SUMMARY_FILE="${COMPARE_OUT}/summary.txt"
{
    echo "======================================================================"
    echo " 吴语专用 ASR 三方对比（test wuyu 151 条）"
    echo "======================================================================"
    if [[ "${RUN_CTRL}" == "1" ]]; then
        printf "%-16s  %-8s  %-10s  %s\n" "model" "CER(%)" "Δ(vs Ctrl)" "best_ckpt"
        printf "%-16s  %-8s  %-10s  %s\n" "----------------" "--------" "----------" "---------"
        DELTA_S1=$(compute_delta "${STAGE1_CER}" "${CTRL_CER}")
        DELTA_S2=$(compute_delta "${STAGE2_CER}" "${CTRL_CER}")
        printf "%-16s  %-8s  %-10s  %s\n" "stage1-only" "${STAGE1_CER}" "${DELTA_S1}" "${STAGE1_BEST_TXT}"
        printf "%-16s  %-8s  %-10s  %s\n" "stage2 (2-stage)" "${STAGE2_CER}" "${DELTA_S2}" "${STAGE2_BEST_TXT}"
        printf "%-16s  %-8s  %-10s  %s\n" "control (base)" "${CTRL_CER}" "+0.00" "${CTRL_BEST_TXT}"
    else
        printf "%-16s  %-8s  %s\n" "model" "CER(%)" "best_ckpt"
        printf "%-16s  %-8s  %s\n" "----------------" "--------" "---------"
        printf "%-16s  %-8s  %s\n" "stage1-only" "${STAGE1_CER}" "${STAGE1_BEST_TXT}"
        printf "%-16s  %-8s  %s\n" "stage2 (2-stage)" "${STAGE2_CER}" "${STAGE2_BEST_TXT}"
    fi
    echo "======================================================================"
} | tee "${SUMMARY_FILE}"

# =============================================================================
# 最终总结
# =============================================================================
echo
echo "============================================================"
echo "[run_wuyu_pipeline] ★ pipeline 完成"
echo "------------------------------------------------------------"
echo "  Stage 1 best      : ${STAGE1_BEST_TXT}"
echo "  Stage 2 best      : ${STAGE2_BEST_TXT}"
if [[ "${RUN_CTRL}" == "1" ]]; then
    echo "  Control best      : ${CTRL_BEST_TXT}"
fi
echo "------------------------------------------------------------"
echo "  Stage 1 pred      : ${STAGE1_OUT}/pred_wuyu_test.jsonl"
echo "  Stage 2 pred      : ${STAGE2_OUT}/pred_wuyu_test.jsonl"
if [[ "${RUN_CTRL}" == "1" ]]; then
    echo "  Control pred      : ${CTRL_OUT}/pred_wuyu_test.jsonl"
fi
echo "------------------------------------------------------------"
echo "  Stage 1 CER file  : ${STAGE1_OUT}/wer_eval/result.wer"
echo "  Stage 2 CER file  : ${STAGE2_OUT}/wer_eval/result.wer"
if [[ "${RUN_CTRL}" == "1" ]]; then
    echo "  Control CER file  : ${CTRL_OUT}/wer_eval/result.wer"
fi
echo "  Summary           : ${SUMMARY_FILE}"
echo "============================================================"

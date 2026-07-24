#!/usr/bin/env bash
# =============================================================================
# hpo_utils.sh
# -----------------------------------------------------------------------------
# VC_v2 HPO 通用辅助函数库。被 run_hpo.sh 通过 `source` 引入。
#
# 提供的函数：
#   trial_name(lr, bs, wu, sched, esp, ep, seed)   -> stdout: trial 目录名
#   trial_dir(name)                                -> stdout: 绝对路径
#   is_done(trial_dir)                             -> return 0 表示 wer_eval/dialect_accuracy.txt 已存在
#   need_eval_only(trial_dir)                      -> return 0 表示训练已完成但评估未完成
#   parse_lid_acc(trial_dir)                       -> stdout: "0.834414"（无换行）
#   parse_overall_cer(trial_dir)                   -> stdout: "0.1234"（无换行）
#   parse_final_epoch(trial_dir)                   -> stdout: 该 trial 最终训练 epoch（浮点数）
#   log_info / log_warn / log_err                  -> 统一带时间戳的日志
#   run_trial(...)                                 -> 完整跑一个 trial（训练+评估）并追加到 summary.csv
#
# 前置约定：
#   - 由 run_hpo.sh 提供 HPO_ROOT / SUMMARY_CSV / STATE_JSON 等环境变量
#   - 使用 bash 4+
# =============================================================================

set -euo pipefail

# ---------- 日志 ----------
log_info() { echo "[$(date '+%F %T')][INFO ] $*" >&2; }
log_warn() { echo "[$(date '+%F %T')][WARN ] $*" >&2; }
log_err()  { echo "[$(date '+%F %T')][ERROR] $*" >&2; }

# ---------- 命名 & 路径 ----------
# trial_name lr bs wu sched esp ep seed
trial_name() {
    local lr="$1" bs="$2" wu="$3" sched="$4" esp="$5" ep="$6" seed="$7"
    echo "trial_lr${lr}_bs${bs}_wu${wu}_sched-${sched}_esp${esp}_ep${ep}_seed${seed}"
}

trial_dir() {
    local name="$1"
    echo "${HPO_ROOT}/${name}"
}

# ---------- 状态判断 ----------
# 已完成（有 lid 分数）
is_done() {
    local td="$1"
    [[ -f "${td}/wer_eval/dialect_accuracy.txt" ]] && \
        grep -q "^accuracy:" "${td}/wer_eval/dialect_accuracy.txt"
}

# 训练结束（有 best_ckpt.txt）但评估未完成
need_eval_only() {
    local td="$1"
    [[ -f "${td}/best_ckpt.txt" ]] && ! is_done "${td}"
}

# ---------- 结果解析 ----------
# 从 wer_eval/dialect_accuracy.txt 抓 "accuracy: 0.xxxxxx (xx.xx%)"
parse_lid_acc() {
    local td="$1"
    local f="${td}/wer_eval/dialect_accuracy.txt"
    if [[ ! -f "${f}" ]]; then
        echo "NaN"; return
    fi
    awk '/^accuracy:/ {print $2; exit}' "${f}"
}

# 从 wer_eval/result.wer 抓整体 CER；文件末尾有形如：
#   Overall -> 14.68 % N=41424 C=35619 S=5445 D=360 I=276
# 解析：抓出百分数并归一化到 [0,1] 小数
parse_overall_cer() {
    local td="$1"
    local f="${td}/wer_eval/result.wer"
    if [[ ! -f "${f}" ]]; then
        echo "NaN"; return
    fi
    python3 - "${f}" <<'PY' 2>/dev/null || echo "NaN"
import re, sys, pathlib
txt = pathlib.Path(sys.argv[1]).read_text(errors="ignore")
# 1) 优先抓 "Overall -> XX.XX %" 或 "Overall -> XX.XX%"（不区分大小写）
m = re.search(r"(?im)^\s*Overall\s*->\s*([0-9]+(?:\.[0-9]+)?)\s*%", txt)
if m:
    print(f"{float(m.group(1)) / 100.0:.6f}")
    sys.exit(0)
# 2) 兼容 "WER: 0.1234" / "CER: 0.1234" 单独一行的写法
for line in txt.splitlines():
    line = line.strip()
    m2 = re.match(r"(?i)^(?:overall\s+)?(?:wer|cer)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?", line)
    if m2:
        v = float(m2.group(1))
        if v > 1.0 or "%" in line:
            v = v / 100.0
        print(f"{v:.6f}")
        sys.exit(0)
print("NaN")
PY
}

# 从 trainer_state.json 抓最终 epoch
parse_final_epoch() {
    local td="$1"
    # best_ckpt.txt 可能指向 <output_dir>/checkpoint-XXX
    local best=""
    if [[ -f "${td}/best_ckpt.txt" ]]; then
        best="$(head -n1 "${td}/best_ckpt.txt" | tr -d '[:space:]')"
    fi
    # 优先用 output_dir 根目录的 trainer_state.json；缺则用 best_ckpt 下的
    local ts=""
    for cand in "${td}/trainer_state.json" "${best}/trainer_state.json"; do
        if [[ -n "${cand}" && -f "${cand}" ]]; then
            ts="${cand}"; break
        fi
    done
    if [[ -z "${ts}" ]]; then
        # 兜底：output_dir 下最新的 checkpoint-*/trainer_state.json
        ts="$(ls -1t "${td}"/checkpoint-*/trainer_state.json 2>/dev/null | head -n1 || true)"
    fi
    if [[ -z "${ts}" || ! -f "${ts}" ]]; then
        echo "NaN"; return
    fi
    python3 - "${ts}" <<'PY' 2>/dev/null || echo "NaN"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{float(d.get('epoch', 0.0)):.4f}")
PY
}

# ---------- 写 summary.csv ----------
# 头：stage,trial_name,lr,batch_size,warmup_ratio,lr_scheduler,early_stop_patience,epochs,seed,best_ckpt,lid_acc,overall_cer,final_epoch,status,elapsed_sec
summary_header() {
    echo "stage,trial_name,lr,batch_size,warmup_ratio,lr_scheduler,early_stop_patience,epochs,seed,best_ckpt,lid_acc,overall_cer,final_epoch,status,elapsed_sec"
}

ensure_summary_csv() {
    if [[ ! -f "${SUMMARY_CSV}" ]]; then
        summary_header > "${SUMMARY_CSV}"
    fi
}

# append_summary stage trial_name lr bs wu sched esp ep seed best_ckpt lid cer final_epoch status elapsed
append_summary() {
    ensure_summary_csv
    # 用引号包住 best_ckpt 避免路径含逗号
    local stage="$1" name="$2" lr="$3" bs="$4" wu="$5" sched="$6" esp="$7" ep="$8" seed="$9"
    shift 9
    local best="$1" lid="$2" cer="$3" fep="$4" status="$5" elapsed="$6"
    echo "${stage},${name},${lr},${bs},${wu},${sched},${esp},${ep},${seed},\"${best}\",${lid},${cer},${fep},${status},${elapsed}" >> "${SUMMARY_CSV}"
}

# ---------- 单 trial 训练 + 评估 ----------
# run_trial stage lr bs wu sched esp ep seed
run_trial() {
    local stage="$1" lr="$2" bs="$3" wu="$4" sched="$5" esp="$6" ep="$7" seed="$8"
    local name td
    name="$(trial_name "${lr}" "${bs}" "${wu}" "${sched}" "${esp}" "${ep}" "${seed}")"
    td="$(trial_dir "${name}")"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        mkdir -p "${td}"
    fi

    log_info "==== [Stage ${stage}] trial=${name}"
    log_info "     dir=${td}"

    # ---- baseline 完全一致 → 直接复用 outputs_vc_v2 结果 ----
    if [[ "${FORCE_RETRAIN:-0}" != "1" \
            && "${lr}"    == "${HPO_BASE_LR:-2e-5}" \
            && "${bs}"    == "${HPO_BASE_BS:-8}" \
            && "${wu}"    == "${HPO_BASE_WU:-0.03}" \
            && "${sched}" == "${HPO_BASE_SCHED:-cosine}" \
            && "${esp}"   == "${HPO_BASE_ESP:-3}" \
            && "${ep}"    == "${HPO_BASE_EP:-5}" \
            && "${seed}"  == "${HPO_BASE_SEED:-42}" ]] \
        && is_done "${HPO_BASELINE_DIR:-/dev/null}"; then
        local lid cer fep
        lid="$(parse_lid_acc "${HPO_BASELINE_DIR}")"
        cer="$(parse_overall_cer "${HPO_BASELINE_DIR}")"
        fep="$(parse_final_epoch "${HPO_BASELINE_DIR}")"
        log_info "     [reuse-baseline] lid=${lid} cer=${cer} epoch=${fep}"
        if ! grep -q ",${name}," "${SUMMARY_CSV}" 2>/dev/null; then
            append_summary "${stage}" "${name}" "${lr}" "${bs}" "${wu}" "${sched}" "${esp}" "${ep}" "${seed}" \
                "$(cat "${HPO_BASELINE_DIR}/best_ckpt.txt" 2>/dev/null | tr -d '\n' || echo '')" \
                "${lid}" "${cer}" "${fep}" "reused_baseline" "0"
        fi
        echo "${lid}"
        return 0
    fi

    # 已完成 → 直接读旧分
    if [[ "${FORCE_RETRAIN:-0}" != "1" ]] && is_done "${td}"; then
        local lid cer fep
        lid="$(parse_lid_acc "${td}")"
        cer="$(parse_overall_cer "${td}")"
        fep="$(parse_final_epoch "${td}")"
        log_info "     [skip] already done, lid=${lid} cer=${cer} epoch=${fep}"
        # 只在 summary.csv 里没有相同 trial_name 时才追加
        if ! grep -q ",${name}," "${SUMMARY_CSV}" 2>/dev/null; then
            append_summary "${stage}" "${name}" "${lr}" "${bs}" "${wu}" "${sched}" "${esp}" "${ep}" "${seed}" \
                "$(cat "${td}/best_ckpt.txt" 2>/dev/null | tr -d '\n' || echo '')" \
                "${lid}" "${cer}" "${fep}" "reused" "0"
        fi
        # 通过 stdout 返回 lid（run_hpo.sh 会用命令替换接住）
        echo "${lid}"
        return 0
    fi

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        log_info "     [DRY_RUN] would train + eval trial: ${name}"
        cat >&2 <<EOF
NUM_GPUS=${HPO_NUM_GPUS} \\
MODEL_PATH="${HPO_MODEL_PATH}" \\
TRAIN_FILE="${HPO_TRAIN_FILE}" \\
EVAL_FILE="${HPO_EVAL_FILE}" \\
OUTPUT_DIR="${td}" \\
EPOCHS=${ep} \\
BATCH_SIZE=${bs} \\
GRAD_ACC=${HPO_GRAD_ACC} \\
LR=${lr} \\
WARMUP_RATIO=${wu} \\
LR_SCHEDULER=${sched} \\
SAVE_STEPS=${HPO_SAVE_STEPS} \\
SAVE_TOTAL_LIMIT=${HPO_SAVE_TOTAL_LIMIT} \\
EARLY_STOP_PATIENCE=${esp} \\
EARLY_STOP_THRESHOLD=${HPO_EARLY_STOP_THRESHOLD} \\
SEED=${seed} \\
BF16=${HPO_BF16} \\
bash ${SCRIPT_DIR_FT}/train_full_ft.sh
EOF
        echo "NaN"
        return 0
    fi

    # ---- 训练 ----
    local t0 t1
    t0=$(date +%s)
    set +e
    (
        set -e
        NUM_GPUS="${HPO_NUM_GPUS}" \
        MODEL_PATH="${HPO_MODEL_PATH}" \
        TRAIN_FILE="${HPO_TRAIN_FILE}" \
        EVAL_FILE="${HPO_EVAL_FILE}" \
        OUTPUT_DIR="${td}" \
        EPOCHS="${ep}" \
        BATCH_SIZE="${bs}" \
        GRAD_ACC="${HPO_GRAD_ACC}" \
        LR="${lr}" \
        WARMUP_RATIO="${wu}" \
        LR_SCHEDULER="${sched}" \
        SAVE_STEPS="${HPO_SAVE_STEPS}" \
        SAVE_TOTAL_LIMIT="${HPO_SAVE_TOTAL_LIMIT}" \
        EARLY_STOP_PATIENCE="${esp}" \
        EARLY_STOP_THRESHOLD="${HPO_EARLY_STOP_THRESHOLD}" \
        SEED="${seed}" \
        BF16="${HPO_BF16}" \
        bash "${SCRIPT_DIR_FT}/train_full_ft.sh" \
            > "${td}/train.log" 2>&1
    )
    local train_rc=$?
    set -e

    if [[ ${train_rc} -ne 0 ]]; then
        t1=$(date +%s)
        log_err "     training failed (rc=${train_rc}), see ${td}/train.log"
        append_summary "${stage}" "${name}" "${lr}" "${bs}" "${wu}" "${sched}" "${esp}" "${ep}" "${seed}" \
            "" "NaN" "NaN" "NaN" "failed_train" "$((t1 - t0))"
        echo "NaN"
        return 0
    fi

    # ---- 评估 ----
    local pred="${td}/pred_test.jsonl"
    local wer="${td}/wer_eval"
    set +e
    (
        set -e
        OUTPUT_DIR="${td}" \
        DATA_TEST="${HPO_DATA_TEST}" \
        PRED_JSONL="${pred}" \
        WER_DIR="${wer}" \
        NUM_GPUS="${HPO_NUM_GPUS}" \
        bash "${SCRIPT_DIR_FT}/run_eval.sh" \
            > "${td}/eval.log" 2>&1
    )
    local eval_rc=$?
    set -e
    t1=$(date +%s)

    if [[ ${eval_rc} -ne 0 ]] || ! is_done "${td}"; then
        log_err "     eval failed (rc=${eval_rc}), see ${td}/eval.log"
        append_summary "${stage}" "${name}" "${lr}" "${bs}" "${wu}" "${sched}" "${esp}" "${ep}" "${seed}" \
            "$(cat "${td}/best_ckpt.txt" 2>/dev/null | tr -d '\n' || echo '')" \
            "NaN" "NaN" "$(parse_final_epoch "${td}")" "failed_eval" "$((t1 - t0))"
        echo "NaN"
        return 0
    fi

    local lid cer fep
    lid="$(parse_lid_acc "${td}")"
    cer="$(parse_overall_cer "${td}")"
    fep="$(parse_final_epoch "${td}")"
    log_info "     [ok] lid=${lid} cer=${cer} epoch=${fep} elapsed=$((t1 - t0))s"
    echo "${lid}" > "${td}/lid_acc.txt"
    echo "${cer}" > "${td}/overall_cer.txt"
    echo "$((t1 - t0))" > "${td}/elapsed.txt"

    append_summary "${stage}" "${name}" "${lr}" "${bs}" "${wu}" "${sched}" "${esp}" "${ep}" "${seed}" \
        "$(cat "${td}/best_ckpt.txt" 2>/dev/null | tr -d '\n' || echo '')" \
        "${lid}" "${cer}" "${fep}" "ok" "$((t1 - t0))"
    echo "${lid}"
}

# ---------- 数值比较（LID 越大越好，CER 越小越好 tiebreaker） ----------
# is_better lid_new cer_new lid_old cer_old  -> return 0 表示 new 更好
is_better() {
    local ln="$1" cn="$2" lo="$3" co="$4"
    python3 - "$ln" "$cn" "$lo" "$co" <<'PY'
import sys, math
def f(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else float("-inf")
    except Exception:
        return float("-inf")
def fc(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else float("inf")
    except Exception:
        return float("inf")
ln, cn, lo, co = f(sys.argv[1]), fc(sys.argv[2]), f(sys.argv[3]), fc(sys.argv[4])
if ln > lo:
    sys.exit(0)
if ln < lo:
    sys.exit(1)
# lid 并列 → cer 更低者胜
sys.exit(0 if cn < co else 1)
PY
}

#!/usr/bin/env bash
# =============================================================================
# run_hpo.sh —— VC_v2 基线的 LID 导向坐标下降超参搜索
# -----------------------------------------------------------------------------
# 用法：
#   bash challenge_full_ft/hpo/run_hpo.sh                      # 全流程
#   DRY_RUN=1     bash challenge_full_ft/hpo/run_hpo.sh        # 只打印 CMD
#   STAGES=1,2,7  bash challenge_full_ft/hpo/run_hpo.sh        # 只跑指定 stage
#   FORCE_RETRAIN=1 bash challenge_full_ft/hpo/run_hpo.sh      # 跳过 is_done 检测
#
# 产物：
#   outputs_hpo/vc_v2/
#     trial_*/                       每个 trial 的训练 + 评估目录
#     summary.csv                    全量结果表
#     summary.md                     每阶段对比
#     state.json                     每阶段最优超参 + 累计最优组合
#     best.json                      最终最优组合
#     seed_robustness.txt            Stage 7 输出
#     run_hpo.log                    主日志（由用户 tee 写入）
# =============================================================================

set -euo pipefail

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"                # .../hpo
SCRIPT_DIR_FT="$(cd "${SCRIPT_DIR}/.." && pwd)"                            # .../challenge_full_ft
HPO_ROOT="${SCRIPT_DIR_FT}/outputs_hpo/vc_v2"
SUMMARY_CSV="${HPO_ROOT}/summary.csv"
SUMMARY_MD="${HPO_ROOT}/summary.md"
STATE_JSON="${HPO_ROOT}/state.json"
BEST_JSON="${HPO_ROOT}/best.json"
SEED_ROBUST="${HPO_ROOT}/seed_robustness.txt"

mkdir -p "${HPO_ROOT}"

# ---------- 硬编码保护：拒绝写入已有实验目录 ----------
case "${HPO_ROOT}" in
    */outputs_hpo/vc_v2) : ;;
    *) echo "[FATAL] HPO_ROOT 路径不合法: ${HPO_ROOT}" >&2; exit 1 ;;
esac

# ---------- 引入工具函数 ----------
source "${SCRIPT_DIR}/hpo_utils.sh"

# ---------- 固定训练/数据参数（不参与搜索） ----------
export HPO_ROOT SUMMARY_CSV STATE_JSON SCRIPT_DIR_FT
export HPO_NUM_GPUS="${HPO_NUM_GPUS:-8}"
export HPO_GRAD_ACC="${HPO_GRAD_ACC:-4}"
export HPO_BF16="${HPO_BF16:-1}"
export HPO_SAVE_STEPS="${HPO_SAVE_STEPS:-50}"
export HPO_SAVE_TOTAL_LIMIT="${HPO_SAVE_TOTAL_LIMIT:-2}"
export HPO_EARLY_STOP_THRESHOLD="${HPO_EARLY_STOP_THRESHOLD:-0.0}"
export HPO_MODEL_PATH="${HPO_MODEL_PATH:-/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B}"
export HPO_TRAIN_FILE="${HPO_TRAIN_FILE:-${SCRIPT_DIR_FT}/data_vc_v2/train.jsonl}"
export HPO_EVAL_FILE="${HPO_EVAL_FILE:-${SCRIPT_DIR_FT}/data_vc_v2/val.jsonl}"
export HPO_DATA_TEST="${HPO_DATA_TEST:-${SCRIPT_DIR_FT}/data/test.jsonl}"

# ---------- 搜索空间（★=baseline） ----------
LR_LIST=(1e-5 2e-5 3e-5 5e-5 8e-5)
BS_LIST=(4 8 16 32)                                      # per-GPU；全局 = bs*4*8
WU_LIST=(0.0 0.03 0.06 0.10)
SCHED_LIST=(cosine linear constant_with_warmup)
ESP_LIST=(2 3 5 8)
EP_LIST=(5 8 12)                                         # Stage 6 条件性
SEED_LIST=(42 123 2024 7 3407)

# ---------- Baseline（复用 outputs_vc_v2 现有结果） ----------
BASE_LR=2e-5
BASE_BS=8
BASE_WU=0.03
BASE_SCHED=cosine
BASE_ESP=3
BASE_EP=5
BASE_SEED=42
BASELINE_OUTPUT_DIR="${SCRIPT_DIR_FT}/outputs_vc_v2"

# 供 hpo_utils.sh 使用
export HPO_BASE_LR="${BASE_LR}"
export HPO_BASE_BS="${BASE_BS}"
export HPO_BASE_WU="${BASE_WU}"
export HPO_BASE_SCHED="${BASE_SCHED}"
export HPO_BASE_ESP="${BASE_ESP}"
export HPO_BASE_EP="${BASE_EP}"
export HPO_BASE_SEED="${BASE_SEED}"
export HPO_BASELINE_DIR="${BASELINE_OUTPUT_DIR}"

# ---------- 当前最优（初始 = baseline） ----------
CUR_LR="${BASE_LR}"
CUR_BS="${BASE_BS}"
CUR_WU="${BASE_WU}"
CUR_SCHED="${BASE_SCHED}"
CUR_ESP="${BASE_ESP}"
CUR_EP="${BASE_EP}"
CUR_SEED="${BASE_SEED}"
CUR_LID=""
CUR_CER=""

# ---------- 执行选项 ----------
DRY_RUN="${DRY_RUN:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
STAGES_STR="${STAGES:-1,2,3,4,5,6,7}"
IFS=',' read -r -a STAGES_ARR <<< "${STAGES_STR}"

# 判断某个 stage 是否要执行
should_run_stage() {
    local s="$1"
    local x
    for x in "${STAGES_ARR[@]}"; do
        [[ "${x}" == "${s}" ]] && return 0
    done
    return 1
}

# ---------- 计划展示 ----------
echo "============================================================"
echo "[run_hpo] VC_v2 坐标下降 HPO 搜索"
echo "  HPO_ROOT             = ${HPO_ROOT}"
echo "  HPO_NUM_GPUS         = ${HPO_NUM_GPUS}"
echo "  HPO_TRAIN_FILE       = ${HPO_TRAIN_FILE}"
echo "  HPO_EVAL_FILE        = ${HPO_EVAL_FILE}"
echo "  HPO_DATA_TEST        = ${HPO_DATA_TEST}"
echo "  DRY_RUN              = ${DRY_RUN}"
echo "  FORCE_RETRAIN        = ${FORCE_RETRAIN}"
echo "  STAGES               = ${STAGES_STR}"
echo "------------------------------------------------------------"
echo "  Baseline             : lr=${BASE_LR} bs=${BASE_BS} wu=${BASE_WU}"
echo "                         sched=${BASE_SCHED} esp=${BASE_ESP}"
echo "                         epochs=${BASE_EP} seed=${BASE_SEED}"
echo "                         reused from ${BASELINE_OUTPUT_DIR}"
echo "  Stage 1 LR           : ${LR_LIST[*]}"
echo "  Stage 2 BATCH_SIZE   : ${BS_LIST[*]}   (per-GPU; 全局 = bs*4*8)"
echo "  Stage 3 WARMUP_RATIO : ${WU_LIST[*]}"
echo "  Stage 4 LR_SCHEDULER : ${SCHED_LIST[*]}"
echo "  Stage 5 PATIENCE     : ${ESP_LIST[*]}"
echo "  Stage 6 EPOCHS       : ${EP_LIST[*]}  (conditional)"
echo "  Stage 7 SEED         : ${SEED_LIST[*]}"
echo "============================================================"

if [[ "${DRY_RUN}" != "1" ]]; then
    log_info "5 秒后开始..."; sleep 5
fi

ensure_summary_csv

# ---------- Baseline 复用 ----------
register_baseline() {
    local td="${BASELINE_OUTPUT_DIR}"
    if is_done "${td}"; then
        local lid cer fep
        lid="$(parse_lid_acc "${td}")"
        cer="$(parse_overall_cer "${td}")"
        fep="$(parse_final_epoch "${td}")"
        log_info "[baseline] reused: lid=${lid} cer=${cer} epoch=${fep}"
        CUR_LID="${lid}"; CUR_CER="${cer}"
        if ! grep -q ",trial_baseline_from_outputs_vc_v2," "${SUMMARY_CSV}" 2>/dev/null; then
            append_summary "0" "trial_baseline_from_outputs_vc_v2" \
                "${BASE_LR}" "${BASE_BS}" "${BASE_WU}" "${BASE_SCHED}" \
                "${BASE_ESP}" "${BASE_EP}" "${BASE_SEED}" \
                "$(cat "${td}/best_ckpt.txt" 2>/dev/null | tr -d '\n' || echo '')" \
                "${lid}" "${cer}" "${fep}" "baseline_reused" "0"
        fi
    else
        log_warn "[baseline] outputs_vc_v2 未找到 dialect_accuracy.txt，baseline 分数缺失"
        CUR_LID="NaN"; CUR_CER="NaN"
    fi
}

register_baseline

# ---------- 更新 state.json ----------
write_state() {
    local stage_done="$1"
    python3 - "${STATE_JSON}" "${stage_done}" \
        "${CUR_LR}" "${CUR_BS}" "${CUR_WU}" "${CUR_SCHED}" \
        "${CUR_ESP}" "${CUR_EP}" "${CUR_SEED}" \
        "${CUR_LID}" "${CUR_CER}" <<'PY'
import json, sys
p, sd, lr, bs, wu, sched, esp, ep, sd_ = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]
lid, cer = sys.argv[10], sys.argv[11]
d = {
    "last_stage_done": int(sd),
    "current_best": {
        "lr": lr, "batch_size": int(bs), "warmup_ratio": float(wu),
        "lr_scheduler": sched, "early_stop_patience": int(esp),
        "epochs": int(ep), "seed": int(sd_),
        "lid_acc": lid, "overall_cer": cer,
    },
}
with open(p, "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
PY
    log_info "[state] wrote ${STATE_JSON} (last_stage_done=${stage_done})"
}

# ---------- 通用：从候选中挑最优 ----------
# 参数：stage_id, cur_var_name, values..., 用于挑该维度最优
# 返回：全局变量 CUR_* 已更新
# 具体做法：外部循环调用 run_trial，用 is_better 逐个更新 CUR_LID/CUR_CER/CUR_*
sweep_one_dim() {
    local stage="$1"; shift
    local dim="$1"; shift            # 一个字符串标签："LR"/"BS"/... 只用于打印
    local base_val="$1"; shift        # baseline 值（复用 baseline 分数时使用）
    local -a candidates=("$@")

    log_info ">>>> Stage ${stage}: sweep ${dim}, candidates=(${candidates[*]}) baseline=${base_val}"

    local best_val="${base_val}"
    local best_lid="${CUR_LID}"
    local best_cer="${CUR_CER}"

    local v
    for v in "${candidates[@]}"; do
        # 若与当前 base 相等且已有分数，跳过（复用 baseline）
        if [[ "${v}" == "${base_val}" && "${CUR_LID}" != "" && "${CUR_LID}" != "NaN" ]]; then
            # 检查是否 baseline 或已 sweep 过的分数就是当前 CUR_* 组合
            local cur_dim_val
            case "${dim}" in
                LR)     cur_dim_val="${CUR_LR}" ;;
                BS)     cur_dim_val="${CUR_BS}" ;;
                WU)     cur_dim_val="${CUR_WU}" ;;
                SCHED)  cur_dim_val="${CUR_SCHED}" ;;
                ESP)    cur_dim_val="${CUR_ESP}" ;;
                EP)     cur_dim_val="${CUR_EP}" ;;
                SEED)   cur_dim_val="${CUR_SEED}" ;;
                *) cur_dim_val="" ;;
            esac
            if [[ "${cur_dim_val}" == "${v}" ]]; then
                log_info "     [reuse] ${dim}=${v} 使用当前累计最优分 lid=${CUR_LID} cer=${CUR_CER}"
                continue
            fi
        fi

        # 组装参数（当前维度用 v，其它维度用 CUR_*）
        local lr="${CUR_LR}" bs="${CUR_BS}" wu="${CUR_WU}" sched="${CUR_SCHED}"
        local esp="${CUR_ESP}" ep="${CUR_EP}" seed="${CUR_SEED}"
        case "${dim}" in
            LR)     lr="${v}" ;;
            BS)     bs="${v}" ;;
            WU)     wu="${v}" ;;
            SCHED)  sched="${v}" ;;
            ESP)    esp="${v}" ;;
            EP)     ep="${v}" ;;
            SEED)   seed="${v}" ;;
        esac

        local lid_out
        lid_out="$(run_trial "${stage}" "${lr}" "${bs}" "${wu}" "${sched}" "${esp}" "${ep}" "${seed}" | tail -n1)"
        local cer_out fep_out
        local name td
        name="$(trial_name "${lr}" "${bs}" "${wu}" "${sched}" "${esp}" "${ep}" "${seed}")"
        td="$(trial_dir "${name}")"
        cer_out="$(parse_overall_cer "${td}")"
        fep_out="$(parse_final_epoch "${td}")"

        if is_better "${lid_out}" "${cer_out}" "${best_lid}" "${best_cer}"; then
            best_val="${v}"
            best_lid="${lid_out}"
            best_cer="${cer_out}"
            log_info "     [★] new best on stage ${stage}: ${dim}=${v} lid=${lid_out} cer=${cer_out}"
        fi
    done

    # 把维度最优写回 CUR_*
    case "${dim}" in
        LR)     CUR_LR="${best_val}" ;;
        BS)     CUR_BS="${best_val}" ;;
        WU)     CUR_WU="${best_val}" ;;
        SCHED)  CUR_SCHED="${best_val}" ;;
        ESP)    CUR_ESP="${best_val}" ;;
        EP)     CUR_EP="${best_val}" ;;
        SEED)   CUR_SEED="${best_val}" ;;
    esac
    CUR_LID="${best_lid}"
    CUR_CER="${best_cer}"
    log_info "<<<< Stage ${stage} done: best ${dim}=${best_val} (lid=${best_lid}, cer=${best_cer})"
    write_state "${stage}"
}

# ---------- Stage 1: LR ----------
if should_run_stage 1; then
    sweep_one_dim 1 LR "${BASE_LR}" "${LR_LIST[@]}"
fi

# ---------- Stage 2: BATCH_SIZE ----------
if should_run_stage 2; then
    sweep_one_dim 2 BS "${BASE_BS}" "${BS_LIST[@]}"
fi

# ---------- Stage 3: WARMUP_RATIO ----------
if should_run_stage 3; then
    sweep_one_dim 3 WU "${BASE_WU}" "${WU_LIST[@]}"
fi

# ---------- Stage 4: LR_SCHEDULER ----------
if should_run_stage 4; then
    sweep_one_dim 4 SCHED "${BASE_SCHED}" "${SCHED_LIST[@]}"
fi

# ---------- Stage 5: EARLY_STOP_PATIENCE ----------
STAGE5_FINAL_EPOCH="NaN"
if should_run_stage 5; then
    sweep_one_dim 5 ESP "${BASE_ESP}" "${ESP_LIST[@]}"
    # 记录 Stage 5 最优 trial 的 final_epoch 供 Stage 6 判定
    STAGE5_NAME="$(trial_name "${CUR_LR}" "${CUR_BS}" "${CUR_WU}" "${CUR_SCHED}" "${CUR_ESP}" "${CUR_EP}" "${CUR_SEED}")"
    STAGE5_TD="$(trial_dir "${STAGE5_NAME}")"
    if is_done "${STAGE5_TD}"; then
        STAGE5_FINAL_EPOCH="$(parse_final_epoch "${STAGE5_TD}")"
    elif [[ "${CUR_LR}" == "${BASE_LR}" \
            && "${CUR_BS}" == "${BASE_BS}" \
            && "${CUR_WU}" == "${BASE_WU}" \
            && "${CUR_SCHED}" == "${BASE_SCHED}" \
            && "${CUR_ESP}" == "${BASE_ESP}" \
            && "${CUR_EP}" == "${BASE_EP}" \
            && "${CUR_SEED}" == "${BASE_SEED}" ]] \
         && is_done "${BASELINE_OUTPUT_DIR}"; then
        STAGE5_FINAL_EPOCH="$(parse_final_epoch "${BASELINE_OUTPUT_DIR}")"
    fi
    log_info "[stage5-final-epoch] ${STAGE5_FINAL_EPOCH}"
fi

# ---------- Stage 6: EPOCHS（条件性） ----------
if should_run_stage 6; then
    # 判定：若 STAGE5_FINAL_EPOCH >= CUR_EP - 0.5，说明训练跑满，早停未触发 → 执行
    trigger6="$(python3 - "${STAGE5_FINAL_EPOCH}" "${CUR_EP}" <<'PY'
import sys, math
fe = sys.argv[1]; ep = sys.argv[2]
try:
    fe_v = float(fe); ep_v = float(ep)
    if math.isnan(fe_v):
        print("skip"); sys.exit()
    print("run" if fe_v >= ep_v - 0.5 else "skip")
except Exception:
    print("skip")
PY
)"
    if [[ "${trigger6}" == "run" ]]; then
        log_info "[stage6] early stopping NOT triggered at epoch=${STAGE5_FINAL_EPOCH}, sweeping EPOCHS"
        sweep_one_dim 6 EP "${BASE_EP}" "${EP_LIST[@]}"
    else
        log_info "[stage6] SKIPPED: early stopping triggered at epoch=${STAGE5_FINAL_EPOCH} (<${CUR_EP}-0.5)"
        write_state 6
    fi
fi

# ---------- Stage 7: SEED 鲁棒性 ----------
if should_run_stage 7; then
    log_info ">>>> Stage 7: SEED robustness, candidates=(${SEED_LIST[*]}) base_seed=${BASE_SEED}"
    # 收集所有 seed 的 lid 分数
    declare -a SEED_LIDS=()
    declare -a SEED_CERS=()
    declare -a SEED_VALS=()
    for sd in "${SEED_LIST[@]}"; do
        lid_out="$(run_trial 7 "${CUR_LR}" "${CUR_BS}" "${CUR_WU}" "${CUR_SCHED}" "${CUR_ESP}" "${CUR_EP}" "${sd}" | tail -n1)"
        name="$(trial_name "${CUR_LR}" "${CUR_BS}" "${CUR_WU}" "${CUR_SCHED}" "${CUR_ESP}" "${CUR_EP}" "${sd}")"
        td="$(trial_dir "${name}")"
        cer_out="$(parse_overall_cer "${td}")"
        SEED_LIDS+=("${lid_out}")
        SEED_CERS+=("${cer_out}")
        SEED_VALS+=("${sd}")
    done

    # 挑最优 seed & 计算 mean/std
    python3 - "${SEED_ROBUST}" "${SEED_VALS[*]}" "${SEED_LIDS[*]}" "${SEED_CERS[*]}" <<'PY' > "${HPO_ROOT}/.best_seed.txt"
import sys, math, statistics as st
out_path = sys.argv[1]
seeds = sys.argv[2].split()
lids  = sys.argv[3].split()
cers  = sys.argv[4].split()

def f(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except Exception:
        return None

pairs = []
for s, l, c in zip(seeds, lids, cers):
    lv = f(l); cv = f(c)
    if lv is None: continue
    pairs.append((s, lv, cv if cv is not None else float("inf")))

# 排序：lid desc, cer asc
pairs.sort(key=lambda x: (-x[1], x[2]))
best = pairs[0] if pairs else (None, None, None)

lid_vals = [p[1] for p in pairs]
mean = st.mean(lid_vals) if lid_vals else float("nan")
sd_  = st.pstdev(lid_vals) if len(lid_vals) > 1 else 0.0

with open(out_path, "w") as f_:
    f_.write("SEED robustness (Stage 7)\n")
    f_.write("=========================\n")
    for s, l, c in sorted(pairs, key=lambda x: seeds.index(x[0])):
        f_.write(f"seed={s:>6}  lid={l:.6f}  cer={c:.6f}\n")
    f_.write("-------------------------\n")
    f_.write(f"mean(lid) = {mean:.6f}\n")
    f_.write(f"std(lid)  = {sd_:.6f}\n")
    f_.write(f"best_seed = {best[0]} (lid={best[1]:.6f}, cer={best[2] if math.isfinite(best[2]) else 'NaN'})\n")

# 输出 best_seed 到 stdout 供 shell 抓取
print(best[0] if best[0] is not None else "")
PY
    BEST_SEED="$(cat "${HPO_ROOT}/.best_seed.txt" | tr -d '[:space:]')"
    if [[ -n "${BEST_SEED}" ]]; then
        CUR_SEED="${BEST_SEED}"
        # 更新 CUR_LID / CUR_CER 到 best seed 那个 trial
        name="$(trial_name "${CUR_LR}" "${CUR_BS}" "${CUR_WU}" "${CUR_SCHED}" "${CUR_ESP}" "${CUR_EP}" "${CUR_SEED}")"
        td="$(trial_dir "${name}")"
        CUR_LID="$(parse_lid_acc "${td}")"
        CUR_CER="$(parse_overall_cer "${td}")"
    fi
    log_info "<<<< Stage 7 done: best_seed=${CUR_SEED} (lid=${CUR_LID}, cer=${CUR_CER})"
    log_info "     robustness report -> ${SEED_ROBUST}"
    write_state 7
fi

# ---------- 生成 summary.md ----------
python3 - "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv, sys
from collections import defaultdict
csv_p, md_p = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_p)))
# group by stage
groups = defaultdict(list)
for r in rows:
    groups[r["stage"]].append(r)

def fnum(x):
    try: return float(x)
    except: return float("nan")

with open(md_p, "w") as f:
    f.write("# VC_v2 HPO Summary\n\n")
    for stage in sorted(groups.keys(), key=lambda x: int(x) if x.isdigit() else -1):
        gs = groups[stage]
        f.write(f"## Stage {stage}\n\n")
        f.write("| trial | lr | bs(global) | wu | sched | esp | ep | seed | lid_acc | overall_cer | final_epoch | status |\n")
        f.write("|-------|----|-----------|----|-------|-----|----|------|---------|-------------|-------------|--------|\n")
        gs_sorted = sorted(gs, key=lambda r: (-fnum(r["lid_acc"]), fnum(r["overall_cer"])))
        best_name = gs_sorted[0]["trial_name"] if gs_sorted else None
        for r in gs:
            bs = int(r["batch_size"]) if r["batch_size"].isdigit() else r["batch_size"]
            try: g_bs = int(bs) * 4 * 8
            except: g_bs = "?"
            mark = " ★" if r["trial_name"] == best_name else ""
            f.write(f"| {r['trial_name']}{mark} | {r['lr']} | {bs}({g_bs}) | {r['warmup_ratio']} | {r['lr_scheduler']} | {r['early_stop_patience']} | {r['epochs']} | {r['seed']} | {r['lid_acc']} | {r['overall_cer']} | {r['final_epoch']} | {r['status']} |\n")
        f.write("\n")
PY
log_info "[summary] wrote ${SUMMARY_MD}"

# ---------- 生成 best.json ----------
python3 - "${BEST_JSON}" \
    "${CUR_LR}" "${CUR_BS}" "${CUR_WU}" "${CUR_SCHED}" \
    "${CUR_ESP}" "${CUR_EP}" "${CUR_SEED}" \
    "${CUR_LID}" "${CUR_CER}" "${HPO_ROOT}" <<'PY'
import json, os, sys, glob

p = sys.argv[1]
lr, bs, wu, sched, esp, ep, seed = sys.argv[2:9]
lid, cer = sys.argv[9], sys.argv[10]
hpo_root = sys.argv[11]

# 定位 best_ckpt
name = f"trial_lr{lr}_bs{bs}_wu{wu}_sched-{sched}_esp{esp}_ep{ep}_seed{seed}"
td = os.path.join(hpo_root, name)
best_ckpt = ""
best_txt = os.path.join(td, "best_ckpt.txt")
if os.path.isfile(best_txt):
    best_ckpt = open(best_txt).read().strip()
elif not os.path.isdir(td):
    # 可能是 baseline 复用
    b = os.path.join(os.path.dirname(hpo_root.rstrip('/')), "..", "outputs_vc_v2", "best_ckpt.txt")
    b = os.path.abspath(b)
    if os.path.isfile(b):
        best_ckpt = open(b).read().strip()

d = {
    "lr": lr,
    "batch_size": int(bs),
    "global_batch_size": int(bs) * 4 * 8,
    "warmup_ratio": float(wu),
    "lr_scheduler": sched,
    "early_stop_patience": int(esp),
    "epochs": int(ep),
    "seed": int(seed),
    "best_ckpt": best_ckpt,
    "lid_acc": lid,
    "overall_cer": cer,
    "trial_name": name,
    "trial_dir": td,
}
with open(p, "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print(f"[best.json] wrote {p}")
PY

# ---------- 最终打印 ----------
echo "============================================================"
echo "[run_hpo] ★ 搜索完成"
echo "  Baseline    : lid=$(grep -E '^0,trial_baseline' "${SUMMARY_CSV}" | awk -F, '{print $11}' | head -n1) (from outputs_vc_v2)"
echo "  Best        : lid=${CUR_LID} cer=${CUR_CER}"
echo "                lr=${CUR_LR} bs=${CUR_BS}(global=$((CUR_BS * 4 * 8)))"
echo "                wu=${CUR_WU} sched=${CUR_SCHED}"
echo "                esp=${CUR_ESP} epochs=${CUR_EP} seed=${CUR_SEED}"
echo "  Artifacts   :"
echo "    ${SUMMARY_CSV}"
echo "    ${SUMMARY_MD}"
echo "    ${STATE_JSON}"
echo "    ${BEST_JSON}"
[[ -f "${SEED_ROBUST}" ]] && echo "    ${SEED_ROBUST}"
echo "============================================================"

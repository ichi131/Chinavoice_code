#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 run_lid.sh 里两个 step 的 LID 解码结果融合成最终提交用的 lid.jsonl：
  - FR_INFER : FireRedLID 在 evaluation_set 上的解码结果（run_lid.sh step1 产出）
  - QW_INFER : Qwen3ASR-LID 在 evaluation_set 上的解码结果（run_lid.sh step2 的
               result.jsonl，取 utt_id/pred_dialect 字段）

融合策略（策略 C）：两模型预测一致 -> 直接采用；不一致 -> 按 FR_DEV/QW_DEV（带
参考答案的 dev/test 集解码结果）算出的 per-label precision 择优，平局倾向 FR。
FR_DEV/QW_DEV 是可选项：不提供（默认）或文件不存在时，直接跳过 precision 表，
不一致时固定采用 FR（对应 precision 表全空时 fs=qs=0.0 的平局默认行为）。
"""

import json
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 路径配置（均可用环境变量覆盖） ============
# dev/test 集解码结果（可选，仅用于按 label precision 择优；无则跳过）
FR_DEV = os.environ.get("FR_DEV", "")
QW_DEV = os.environ.get("QW_DEV", "")

# evaluation_set 上两个 step 的解码结果（run_lid.sh 里实际产出的路径）
FR_INFER = os.environ.get(
    "FR_INFER",
    os.path.join(REPO_ROOT, "FireRedASR2S-fintuning/exp/lid_output/evaluation_pred.jsonl"),
)
QW_INFER = os.environ.get(
    "QW_INFER",
    os.path.join(REPO_ROOT, "challenge_full_ft/infer_data/result.jsonl"),
)

OUT_PATH = os.environ.get("OUT_PATH", os.path.join(REPO_ROOT, "infer_data/lid.jsonl"))


def load_precision_table():
    if not FR_DEV or not QW_DEV or not os.path.isfile(FR_DEV) or not os.path.isfile(QW_DEV):
        print("[apply_fuse_c_v5] FR_DEV/QW_DEV 未提供或文件不存在，跳过 precision 表，"
              "不一致时默认采用 FR")
        return {}, {}, {}, {}

    fr_pred, fr_ref = {}, {}
    for line in open(FR_DEV):
        d = json.loads(line)
        fr_pred[d["key"]] = d["accent"]
        fr_ref[d["key"]] = d["wav_path"].split("/")[-3]

    qw_pred, qw_ref = {}, {}
    for line in open(QW_DEV):
        d = json.loads(line)
        qw_pred[d["utt_id"]] = d["pred_dialect"]
        qw_ref[d["utt_id"]] = d["ref_dialect"]

    fr_prec = defaultdict(lambda: [0, 0])
    for k, p in fr_pred.items():
        r = fr_ref.get(k)
        if r is None:
            continue
        fr_prec[p][0] += 1
        if p == r:
            fr_prec[p][1] += 1

    qw_prec = defaultdict(lambda: [0, 0])
    for k, p in qw_pred.items():
        r = qw_ref.get(k)
        if r is None:
            continue
        qw_prec[p][0] += 1
        if p == r:
            qw_prec[p][1] += 1

    fr_p = {lab: (v[1] / v[0] if v[0] else 0.0) for lab, v in fr_prec.items()}
    qw_p = {lab: (v[1] / v[0] if v[0] else 0.0) for lab, v in qw_prec.items()}
    return fr_p, qw_p, fr_prec, qw_prec


def load_infer():
    fr = {}
    for line in open(FR_INFER):
        d = json.loads(line)
        fr[d["key"]] = d["accent"]

    qw = {}
    for line in open(QW_INFER):
        d = json.loads(line)
        qw[d["utt_id"]] = d["pred_dialect"]
    return fr, qw


def fuse(fp, qp, fr_p, qw_p):
    if fp == qp:
        return fp, "agree"
    fs = fr_p.get(fp)
    qs = qw_p.get(qp)
    if fs is None:
        fs = sum(fr_p.values()) / max(len(fr_p), 1)
    if qs is None:
        qs = sum(qw_p.values()) / max(len(qw_p), 1)
    if fs >= qs:
        return fp, f"FR(fp_prec={fs:.4f} >= qp_prec={qs:.4f})"
    else:
        return qp, f"QW(qp_prec={qs:.4f} > fp_prec={fs:.4f})"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    fr_p, qw_p, fr_prec_raw, qw_prec_raw = load_precision_table()
    print("[Precision-by-predicted-label 表 (来自 dev)]")
    print(f"{'label':<12}{'FR_prec':>10}{'FR_n':>7}{'QW_prec':>10}{'QW_n':>7}")
    all_labels = sorted(set(fr_p) | set(qw_p))
    for lab in all_labels:
        fp_ = fr_p.get(lab, float("nan"))
        qp_ = qw_p.get(lab, float("nan"))
        fn_ = fr_prec_raw.get(lab, [0, 0])[0]
        qn_ = qw_prec_raw.get(lab, [0, 0])[0]
        print(f"{lab:<12}{fp_:>10.4f}{fn_:>7}{qp_:>10.4f}{qn_:>7}")

    fr, qw = load_infer()
    keys_fr = set(fr)
    keys_qw = set(qw)
    common = keys_fr & keys_qw
    only_fr = keys_fr - keys_qw
    only_qw = keys_qw - keys_fr
    print()
    print(f"FR 推理数量 : {len(fr)}")
    print(f"QW 推理数量 : {len(qw)}")
    print(f"共同 key    : {len(common)}")
    print(f"仅 FR 有    : {len(only_fr)}")
    print(f"仅 QW 有    : {len(only_qw)}")

    # 统计 QW 侧空 dialect 的样本（v5 已知有 2 条）会被融合成什么
    qw_empty_keys = [k for k, v in qw.items() if v == "" or v is None]
    print(f"QW 侧空 dialect key: {len(qw_empty_keys)} 个 -> {qw_empty_keys}")

    agree = 0
    disagree = 0
    take_fr = 0
    take_qw = 0

    order = []
    seen = set()
    for line in open(FR_INFER):
        k = json.loads(line)["key"]
        if k not in seen:
            order.append(k)
            seen.add(k)

    overridden_empty = []  # 记录：QW 空 dialect 被 FR 覆盖的情况

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for k in order:
            fp = fr.get(k)
            qp = qw.get(k)
            if fp is not None and qp is not None:
                fused, reason = fuse(fp, qp, fr_p, qw_p)
                if fp == qp:
                    agree += 1
                else:
                    disagree += 1
                    if fused == fp:
                        take_fr += 1
                    else:
                        take_qw += 1
                if qp == "" or qp is None:
                    overridden_empty.append((k, qp, fp, fused, reason))
            elif fp is not None:
                fused = fp
            else:
                fused = qp
            f.write(json.dumps({"key": k, "dialect": fused}, ensure_ascii=False, separators=(",", ":")) + "\n")

        for k in sorted(only_qw):
            f.write(json.dumps({"key": k, "dialect": qw[k]}, ensure_ascii=False, separators=(",", ":")) + "\n")

    print()
    print("=" * 60)
    print("融合完成")
    print("=" * 60)
    total = agree + disagree
    if total:
        print(f"两模型一致           : {agree}/{total} = {agree/total*100:.2f}%")
        print(f"两模型不一致         : {disagree}/{total} = {disagree/total*100:.2f}%")
        if disagree:
            print(f"  ┣ 采用 FR 预测     : {take_fr}/{disagree} = {take_fr/disagree*100:.2f}%")
            print(f"  ┗ 采用 QW 预测     : {take_qw}/{disagree} = {take_qw/disagree*100:.2f}%")

    if overridden_empty:
        print()
        print(f"[QW 空 dialect 样本融合详情] {len(overridden_empty)} 条")
        for k, qp, fp, fused, reason in overridden_empty:
            print(f"  {k}: QW='' vs FR='{fp}' -> 融合={fused} ({reason})")

    print(f"输出文件             : {OUT_PATH}")


if __name__ == "__main__":
    main()

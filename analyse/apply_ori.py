#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 apply_fuse_c.py 的 v5 版本：
  - Qwen 推理侧输入改为最新的 lid_from_pred.jsonl（由 pred_eval.jsonl 提取）
  - 输出目录改为 infer_data/mix_v5/lid.jsonl
其余融合策略与 apply_fuse_c.py 完全一致（策略 C：一致->保持；不一致->按 dev 集上
预测标签的 precision 择优；平局倾向 FR）。
"""

import json
import os
from collections import defaultdict

# ============ 路径配置 ============
FR_DEV = "data_test_pred_1.jsonl"
QW_DEV = "pred_test.jsonl"

FR_INFER = "/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/workspace/FireRedASR2S-fintuning/exp/lid_chinavoices_data_speaker_ft_encoder/evaluation_pred_1.jsonl"
QW_INFER = "/mnt/geminihzceph/user_johannapeng/challenge_model/infer_data/lid_from_pred.jsonl"

OUT_PATH = "/mnt/geminihzceph/user_johannapeng/challenge_model/infer_data/mix_v5/lid_eval_1.jsonl"


def load_precision_table():
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
        qw[d["key"]] = d["dialect"]
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

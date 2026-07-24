#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_lid_precision.py
========================
从 pred_test.jsonl 计算 LID（方言分类）的**预测置信度**（Precision）等指标。

Precision 的含义：
    当模型预测某样本为标签 L 时，它真的属于 L 的概率 = TP_L / (TP_L + FP_L)
    也就是混淆矩阵中「L 这一列」的对角线值 / 「L 这一列」的总和。

Recall 的含义：
    真实为标签 L 的样本中被正确识别的比例 = TP_L / (TP_L + FN_L)
    也就是混淆矩阵中「L 这一行」的对角线值 / 「L 这一行」的总和。

用法：
    python compute_lid_precision.py \
        --pred_jsonl /path/to/pred_test.jsonl \
        --out /path/to/lid_precision.txt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Compute LID precision / recall / F1 from pred_test.jsonl"
    )
    p.add_argument("--pred_jsonl", required=True, type=str,
                   help="pred_test.jsonl（需包含 ref_dialect 与 pred_dialect 字段）")
    p.add_argument("--out", required=True, type=str,
                   help="输出的报表文件路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 读入并统计混淆矩阵
    tp = defaultdict(int)   # pred == ref
    pred_total = defaultdict(int)  # 该 label 被预测的总次数
    ref_total = defaultdict(int)   # 该 label 真实出现的总次数
    labels_set = set()

    total_evaluable = 0
    skipped = 0
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            ref = obj.get("ref_dialect") or ""
            pred = obj.get("pred_dialect") or ""
            err = obj.get("error") or ""
            if err or not ref or not pred:
                skipped += 1
                continue
            total_evaluable += 1
            labels_set.add(ref)
            labels_set.add(pred)
            ref_total[ref] += 1
            pred_total[pred] += 1
            if pred == ref:
                tp[ref] += 1

    labels = sorted(labels_set)

    # 2. 逐 label 计算 P / R / F1
    per_label = []
    tp_sum = sum(tp.values())
    for lab in labels:
        p = tp[lab] / pred_total[lab] if pred_total[lab] else 0.0
        r = tp[lab] / ref_total[lab] if ref_total[lab] else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label.append((lab, ref_total[lab], pred_total[lab], tp[lab], p, r, f1))

    # 3. 整体聚合
    micro = tp_sum / total_evaluable if total_evaluable else 0.0
    macro_p = sum(x[4] for x in per_label) / len(per_label) if per_label else 0.0
    macro_r = sum(x[5] for x in per_label) / len(per_label) if per_label else 0.0
    macro_f1 = sum(x[6] for x in per_label) / len(per_label) if per_label else 0.0
    weighted_p = (
        sum(x[4] * x[1] for x in per_label) / sum(x[1] for x in per_label)
        if per_label else 0.0
    )
    weighted_f1 = (
        sum(x[6] * x[1] for x in per_label) / sum(x[1] for x in per_label)
        if per_label else 0.0
    )

    # 4. 打印 + 落盘
    lines = []
    lines.append("LID prediction confidence (Precision) report")
    lines.append("=" * 60)
    lines.append(f"pred_jsonl:      {args.pred_jsonl}")
    lines.append(f"total_evaluable: {total_evaluable}")
    lines.append(f"skipped:         {skipped}")
    lines.append("")
    lines.append("Per-label metrics")
    lines.append(
        f"{'dialect':<12} {'ref_n':>6} {'pred_n':>7} {'TP':>5}   "
        f"{'precision':>10} {'recall':>8} {'F1':>8}"
    )
    lines.append("-" * 68)
    for lab, rn, pn, t, p, r, f1 in per_label:
        lines.append(
            f"{lab:<12} {rn:>6} {pn:>7} {t:>5}   "
            f"{p*100:>9.2f}% {r*100:>7.2f}% {f1*100:>7.2f}%"
        )
    lines.append("-" * 68)
    lines.append("")
    lines.append("Overall")
    lines.append(f"  accuracy (micro-avg)    = {micro*100:6.2f}%")
    lines.append(f"  macro-precision         = {macro_p*100:6.2f}%")
    lines.append(f"  macro-recall            = {macro_r*100:6.2f}%")
    lines.append(f"  macro-F1                = {macro_f1*100:6.2f}%")
    lines.append(f"  weighted-precision      = {weighted_p*100:6.2f}%")
    lines.append(f"  weighted-F1             = {weighted_f1*100:6.2f}%")
    lines.append("")
    lines.append(
        "解释：precision 即『被预测为该方言时，它真是该方言的概率』——"
        "这就是你要的 LID 预测置信度。"
    )

    report = "\n".join(lines) + "\n"
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()

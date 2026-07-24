#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_confidence.py
=====================

**方案 A 的分析端脚本**。读入 `infer_test_with_confidence.py` 产出的、
带 dialect_conf 字段的 JSONL，做以下事情：

1. 整体 Precision-Coverage 曲线（不同阈值 τ 下保留多少样本 / 精度多少）
2. 找出"达到 target precision（默认 95%）所需的最小 τ"，以及此时的 coverage
3. 逐方言（按 pred_dialect 分组）做同样分析，输出每类的推荐阈值

输入 JSONL 需含字段：pred_dialect / ref_dialect / dialect_conf。
输出：一份文本报表 + 一份逐样本"是否保留"的 CSV（可选）。

用法：
    python analyze_confidence.py \
        --pred_jsonl outputs_vc_v2/pred_test_conf.jsonl \
        --out_report outputs_vc_v2/wer_eval/confidence_report.txt \
        --target_precision 0.95

也支持 --target_precision 0.99 追极致精度。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
# I/O
# ------------------------------------------------------------------ #
def load_pred(pred_jsonl: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                continue
            if not obj.get("pred_dialect") or not obj.get("ref_dialect"):
                continue
            if "dialect_conf" not in obj:
                continue
            samples.append(obj)
    return samples


# ------------------------------------------------------------------ #
# 阈值扫描核心
# ------------------------------------------------------------------ #
def sweep_precision_coverage(
    items: List[Tuple[float, bool]],
    grid: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """
    输入: [(confidence, correct), ...]
    输出: 每个阈值下的 { threshold, kept, coverage, correct_kept, precision } 列表。

    保留规则: confidence >= threshold 就保留。
    """
    if not items:
        return []
    total = len(items)
    if grid is None:
        grid = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
    rows: List[Dict[str, float]] = []
    for th in grid:
        kept = [(c, ok) for c, ok in items if c >= th]
        n_kept = len(kept)
        n_correct = sum(1 for _, ok in kept if ok)
        precision = n_correct / n_kept if n_kept else 0.0
        coverage = n_kept / total if total else 0.0
        rows.append({
            "threshold": th,
            "kept": n_kept,
            "coverage": coverage,
            "correct_kept": n_correct,
            "precision": precision,
        })
    return rows


def find_min_threshold_for_precision(
    items: List[Tuple[float, bool]],
    target: float,
    min_kept: int = 1,
) -> Optional[Dict[str, float]]:
    """
    在离散阈值网格上找到"能让 precision >= target 的最小 threshold"。
    使用逐样本扫描 (O(n log n))，比 fixed-grid 更精确。

    返回 { threshold, kept, coverage, correct_kept, precision } 或 None（做不到）。
    """
    if not items:
        return None
    total = len(items)
    # 按 confidence 从高到低排序
    sorted_items = sorted(items, key=lambda x: -x[0])
    n_kept = 0
    n_correct = 0
    best: Optional[Dict[str, float]] = None
    # 依次把 top-k 都保留，从 k=1 到 n；找到"满足 P>=target 且 kept 最大"的 k
    # 由于我们要"最小 threshold"（等价于"最大 kept"），从头累加找**满足条件的最大 k**
    for k, (conf, ok) in enumerate(sorted_items, start=1):
        n_kept += 1
        if ok:
            n_correct += 1
        precision = n_correct / n_kept
        if precision >= target and n_kept >= min_kept:
            best = {
                "threshold": conf,  # 该 k 对应的边界置信度
                "kept": n_kept,
                "coverage": n_kept / total,
                "correct_kept": n_correct,
                "precision": precision,
            }
        # 不 early break：我们要"最大 kept"（=最低阈值）
    return best


# ------------------------------------------------------------------ #
# 报表
# ------------------------------------------------------------------ #
def format_row_table(rows: List[Dict[str, float]]) -> List[str]:
    lines = []
    lines.append(f"{'thresh':>7} {'kept':>7} {'cov':>8} {'correct':>7} {'precision':>10}")
    lines.append("-" * 46)
    for r in rows:
        lines.append(
            f"{r['threshold']:>7.3f} {r['kept']:>7d} "
            f"{r['coverage']*100:>7.2f}% {r['correct_kept']:>7d} "
            f"{r['precision']*100:>9.2f}%"
        )
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Analyze Plan-A confidence for selective classification."
    )
    p.add_argument("--pred_jsonl", required=True, type=str,
                   help="infer_test_with_confidence.py 产出的 JSONL")
    p.add_argument("--out_report", required=True, type=str,
                   help="报表输出路径 (txt)")
    p.add_argument("--target_precision", type=float, default=0.95,
                   help="目标精度（默认 0.95）；也会额外输出 0.99 一档参考")
    p.add_argument("--min_kept_per_class", type=int, default=5,
                   help="逐类找最优阈值时，至少保留多少样本才算数（避免只留 1 个刷 100%）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_pred(args.pred_jsonl)
    if not samples:
        raise SystemExit(f"[error] {args.pred_jsonl} 没有可分析的样本")

    total = len(samples)
    correct_all = sum(1 for s in samples if s["pred_dialect"] == s["ref_dialect"])
    base_acc = correct_all / total

    # ------------- 1) 全局曲线 -------------
    global_items: List[Tuple[float, bool]] = [
        (float(s["dialect_conf"]), s["pred_dialect"] == s["ref_dialect"])
        for s in samples
    ]
    global_rows = sweep_precision_coverage(global_items)

    global_p95 = find_min_threshold_for_precision(global_items, args.target_precision)
    global_p99 = find_min_threshold_for_precision(global_items, 0.99)

    # ------------- 2) 按预测方言分组 -------------
    by_pred: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    for s in samples:
        by_pred[s["pred_dialect"]].append(
            (float(s["dialect_conf"]), s["pred_dialect"] == s["ref_dialect"])
        )

    per_class_recommend: List[Dict[str, Any]] = []
    for lab in sorted(by_pred.keys()):
        items = by_pred[lab]
        n_lab = len(items)
        n_correct = sum(1 for _, ok in items if ok)
        base_p = n_correct / n_lab if n_lab else 0.0
        rec = find_min_threshold_for_precision(
            items,
            args.target_precision,
            min_kept=args.min_kept_per_class,
        )
        rec99 = find_min_threshold_for_precision(
            items, 0.99, min_kept=args.min_kept_per_class
        )
        per_class_recommend.append({
            "label":          lab,
            "n":              n_lab,
            "base_precision": base_p,
            "rec":            rec,
            "rec99":          rec99,
        })

    # ------------- 3) confidence 分布快照 -------------
    confs_correct = sorted([c for c, ok in global_items if ok])
    confs_wrong = sorted([c for c, ok in global_items if not ok])
    def quantiles(arr, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        if not arr:
            return {q: float("nan") for q in qs}
        out = {}
        for q in qs:
            idx = min(len(arr) - 1, max(0, int(round(q * (len(arr) - 1)))))
            out[q] = arr[idx]
        return out
    q_correct = quantiles(confs_correct)
    q_wrong = quantiles(confs_wrong)

    # ------------- 4) 组装报表 -------------
    lines: List[str] = []
    lines.append("Plan-A confidence analysis report")
    lines.append("=" * 60)
    lines.append(f"pred_jsonl:         {args.pred_jsonl}")
    lines.append(f"total samples:      {total}")
    lines.append(f"baseline accuracy:  {base_acc*100:.2f}%   (无阈值时的整体精度)")
    lines.append(f"target precision:   {args.target_precision*100:.2f}%")
    lines.append("")

    lines.append("[Section 1] Global precision-coverage sweep")
    lines.extend(format_row_table(global_rows))
    lines.append("")

    lines.append(f"[Section 2] 全局最优阈值 (P >= {args.target_precision*100:.1f}%)")
    if global_p95:
        lines.append(
            f"  threshold >= {global_p95['threshold']:.4f}: "
            f"kept={global_p95['kept']}, coverage={global_p95['coverage']*100:.2f}%, "
            f"precision={global_p95['precision']*100:.2f}%"
        )
    else:
        lines.append(f"  做不到 P >= {args.target_precision*100:.1f}%（数据无法在任何阈值下达标）")
    lines.append("")
    lines.append("[Section 2b] 全局最优阈值 (P >= 99%)")
    if global_p99:
        lines.append(
            f"  threshold >= {global_p99['threshold']:.4f}: "
            f"kept={global_p99['kept']}, coverage={global_p99['coverage']*100:.2f}%, "
            f"precision={global_p99['precision']*100:.2f}%"
        )
    else:
        lines.append("  做不到 P >= 99%")
    lines.append("")

    lines.append("[Section 3] 每类 pred_dialect 推荐阈值")
    lines.append(
        f"{'pred_label':<12} {'n':>5} {'base_P':>8} "
        f"{'τ (P>='+str(int(args.target_precision*100))+'%)':>16} "
        f"{'kept':>6} {'cov':>7} {'P':>7}  |  "
        f"{'τ (P>=99%)':>12} {'kept':>6} {'cov':>7} {'P':>7}"
    )
    lines.append("-" * 105)
    for row in per_class_recommend:
        lab = row["label"]
        n = row["n"]
        base_p = row["base_precision"]
        rec = row["rec"]
        rec99 = row["rec99"]
        seg_a = f"{'--':>16} {'--':>6} {'--':>7} {'--':>7}"
        if rec:
            seg_a = (f"{rec['threshold']:>16.4f} {rec['kept']:>6d} "
                     f"{rec['coverage']*100:>6.2f}% {rec['precision']*100:>6.2f}%")
        seg_b = f"{'--':>12} {'--':>6} {'--':>7} {'--':>7}"
        if rec99:
            seg_b = (f"{rec99['threshold']:>12.4f} {rec99['kept']:>6d} "
                     f"{rec99['coverage']*100:>6.2f}% {rec99['precision']*100:>6.2f}%")
        lines.append(
            f"{lab:<12} {n:>5} {base_p*100:>7.2f}% {seg_a}  |  {seg_b}"
        )
    lines.append("")

    lines.append("[Section 4] confidence 分布分位数（越分开、辨识度越好）")
    def _fmt_q(name, q):
        return (f"{name:<10}"
                + " ".join(f"p{int(k*100):02d}={v:.4f}" for k, v in q.items()))
    lines.append(_fmt_q("correct:", q_correct))
    lines.append(_fmt_q("wrong:  ", q_wrong))
    lines.append("")

    lines.append("[Section 5] 使用建议")
    lines.append("- 若追求数据蒸馏 / 半监督标签：直接卡 Section 3 的 tau (P>=95%)")
    lines.append("- 若追求极致精度 / 只挑最有把握的样本：卡 Section 3 的 tau (P>=99%)")
    lines.append("- 每类 tau 差异大很正常——垃圾桶类 (nanchang / wuyu) 需要更高 tau")
    lines.append("- 若 P>=target 那一列出现 --，说明该类在测试集上从未达到目标精度，")
    lines.append("  意味着 dialect_conf 单指标区分不了它——可尝试再乘 text_avg_logprob。")

    report = "\n".join(lines) + "\n"
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"[done] report saved -> {args.out_report}", flush=True)


if __name__ == "__main__":
    main()

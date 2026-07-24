#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
route_summarize.py
==================
读取 baseline (VC v2) 与 hybrid specialist 的评测目录，产出并列对比 summary.txt。

输入约定
--------
--baseline_eval_dir  <dir>
    期望包含：result.wer / by_dialect_summary.txt / dialect_accuracy.txt
    可选：lid_precision.txt（若缺，则该列显示 "n/a"，不阻断）
--hybrid_eval_dir    <dir>
    同上，且必须包含 lid_precision.txt（由 compute_lid_precision.py 产出）
--routed_jsonl       <path>
    路由后的 pred JSONL；用于统计每条样本走了哪条路径（辅助信息）

输出
----
overall CER、wuyu / kejia / nanchang 三方言的 CER、LID by-ref accuracy、LID by-pred precision，
以及 baseline 与 hybrid 的 Δ 列。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 关注的三个方言（可扩展）
FOCUS_DIALECTS = ["wuyu", "kejia", "nanchang"]


# =============================================================================
# 各类文件解析器
# =============================================================================
RE_OVERALL = re.compile(r"^Overall\s*->\s*([\d.]+)\s*%")

def parse_overall_wer(result_wer_path: Path) -> Optional[float]:
    """从 result.wer 末尾的 'Overall -> XX.XX %' 行取整体 CER%。"""
    if not result_wer_path.is_file():
        return None
    last = None
    with result_wer_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = RE_OVERALL.match(line.strip())
            if m:
                last = float(m.group(1))
    return last


def parse_by_dialect_summary(path: Path) -> Dict[str, Tuple[int, float]]:
    """解析 by_dialect_summary.txt -> {dialect: (samples, wer_pct)}。"""
    out: Dict[str, Tuple[int, float]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("dialect") or line.startswith("---") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            dialect = parts[0]
            try:
                samples = int(parts[1])
            except ValueError:
                continue
            wer_str = parts[2].rstrip("%")
            try:
                wer = float(wer_str)
            except ValueError:
                continue
            out[dialect] = (samples, wer)
    return out


def parse_dialect_accuracy(path: Path) -> Dict[str, Tuple[int, int, float]]:
    """
    解析 dialect_accuracy.txt 中的 "Accuracy by ref_dialect" 段。
    返回 {dialect: (samples, correct, accuracy_pct)}。
    """
    out: Dict[str, Tuple[int, int, float]] = {}
    if not path.is_file():
        return out
    in_section = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("Accuracy by ref_dialect"):
                in_section = True
                continue
            if not in_section:
                continue
            if not line.strip():
                # 遇到空行前允许一个 header + 分隔行 -> 空行结束
                if out:
                    break
                continue
            if line.startswith("ref_dialect") or line.startswith("---"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            dialect = parts[0]
            try:
                samples = int(parts[1])
                correct = int(parts[2])
            except ValueError:
                continue
            acc_str = parts[3].rstrip("%")
            try:
                acc = float(acc_str)
            except ValueError:
                continue
            out[dialect] = (samples, correct, acc)
    return out


RE_LID_ROW = re.compile(
    r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s*$"
)

def parse_lid_precision(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    解析 compute_lid_precision.py 生成的 report。
    返回 {label: {ref_n, pred_n, tp, precision, recall, f1}}。
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = RE_LID_ROW.match(line.rstrip("\n"))
            if not m:
                continue
            label = m.group(1)
            if label in ("dialect",):  # header safety
                continue
            out[label] = {
                "ref_n":     int(m.group(2)),
                "pred_n":    int(m.group(3)),
                "tp":        int(m.group(4)),
                "precision": float(m.group(5)),
                "recall":    float(m.group(6)),
                "f1":        float(m.group(7)),
            }
    return out


def summarize_routed_jsonl(path: Path) -> Dict[str, int]:
    """统计路由决策分布（辅助信息）。"""
    stats: Counter = Counter()
    per_specialist: Counter = Counter()
    fallback = 0
    total = 0
    if not path.is_file():
        return {"total": 0}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            total += 1
            ru = obj.get("route_used") or "vc_v2"
            stats[ru] += 1
            if ru != "vc_v2":
                per_specialist[ru] += 1
            if obj.get("route_fallback"):
                fallback += 1
    result: Dict[str, int] = {"total": total, "fallback": fallback}
    result.update({f"used[{k}]": v for k, v in stats.items()})
    return result


# =============================================================================
# 渲染
# =============================================================================
def _fmt_delta(baseline: Optional[float], hybrid: Optional[float], invert: bool = False) -> str:
    """
    invert=False (CER, 越小越好): Δ = hybrid - baseline，负号（更低）好，用 ↓ 标注
    invert=True  (Accuracy, 越大越好): Δ = hybrid - baseline，正号好，用 ↑ 标注
    """
    if baseline is None or hybrid is None:
        return "   n/a"
    d = hybrid - baseline
    if abs(d) < 1e-6:
        return f"  0.00"
    if invert:
        arrow = "↑" if d > 0 else "↓"
    else:
        arrow = "↓" if d < 0 else "↑"
    return f"{d:+6.2f}{arrow}"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:6.2f}%" if v is not None else "   n/a"


def build_summary(
    baseline_eval_dir: Path,
    hybrid_eval_dir:   Path,
    routed_jsonl:      Path,
    run_tag:           str,
    tau_profile:       str,
) -> str:
    # 解析 baseline
    b_overall = parse_overall_wer(baseline_eval_dir / "result.wer")
    b_bydia   = parse_by_dialect_summary(baseline_eval_dir / "by_dialect_summary.txt")
    b_lidacc  = parse_dialect_accuracy(baseline_eval_dir / "dialect_accuracy.txt")
    b_lidprec = parse_lid_precision(baseline_eval_dir / "lid_precision.txt")

    # 解析 hybrid
    h_overall = parse_overall_wer(hybrid_eval_dir / "result.wer")
    h_bydia   = parse_by_dialect_summary(hybrid_eval_dir / "by_dialect_summary.txt")
    h_lidacc  = parse_dialect_accuracy(hybrid_eval_dir / "dialect_accuracy.txt")
    h_lidprec = parse_lid_precision(hybrid_eval_dir / "lid_precision.txt")

    # 路由统计
    route_stats = summarize_routed_jsonl(routed_jsonl)

    lines: List[str] = []
    lines.append("Hybrid Specialist Routing Summary")
    lines.append("=" * 78)
    lines.append(f"run_tag        : {run_tag}")
    lines.append(f"tau_profile    : {tau_profile}")
    lines.append(f"baseline_dir   : {baseline_eval_dir}")
    lines.append(f"hybrid_dir     : {hybrid_eval_dir}")
    lines.append(f"routed_jsonl   : {routed_jsonl}")
    lines.append("")

    # ---- 路由决策分布 ----
    lines.append("[1] Route decision distribution (from routed_jsonl)")
    lines.append("-" * 78)
    lines.append(f"  total samples : {route_stats.get('total', 0)}")
    for k in sorted(route_stats.keys()):
        if k.startswith("used["):
            lines.append(f"  {k:<28s} = {route_stats[k]}")
    lines.append(f"  fallback (kept vc_v2 due to error) : {route_stats.get('fallback', 0)}")
    lines.append("")

    # ---- Overall CER ----
    lines.append("[2] Overall CER (lower is better)")
    lines.append("-" * 78)
    header = f"  {'metric':<22s} {'baseline':>10s} {'hybrid':>10s} {'delta':>9s}"
    lines.append(header)
    lines.append("-" * 78)
    lines.append(
        f"  {'overall CER':<22s} "
        f"{_fmt_pct(b_overall):>10s} "
        f"{_fmt_pct(h_overall):>10s} "
        f"{_fmt_delta(b_overall, h_overall, invert=False):>9s}"
    )
    lines.append("")

    # ---- 按方言 CER（全部 + 突出关注方言） ----
    all_dialects = sorted(set(b_bydia.keys()) | set(h_bydia.keys()))
    focus_set = set(FOCUS_DIALECTS)
    lines.append("[3] By-dialect CER (lower is better) — ★ = focus dialects (routed)")
    lines.append("-" * 78)
    lines.append(f"  {'dialect':<14s} {'n_ref':>7s}  {'baseline':>10s} {'hybrid':>10s} {'delta':>9s}")
    lines.append("-" * 78)
    for dialect in all_dialects:
        b_n, b_wer = b_bydia.get(dialect, (None, None))
        h_n, h_wer = h_bydia.get(dialect, (None, None))
        n_show = b_n if b_n is not None else (h_n if h_n is not None else 0)
        mark = "★" if dialect in focus_set else " "
        lines.append(
            f" {mark}{dialect:<14s} {n_show:>7d}  "
            f"{_fmt_pct(b_wer):>10s} "
            f"{_fmt_pct(h_wer):>10s} "
            f"{_fmt_delta(b_wer, h_wer, invert=False):>9s}"
        )
    lines.append("")

    # ---- LID by-ref accuracy（focus dialects） ----
    lines.append("[4] LID accuracy by ref_dialect (higher is better) — focus dialects")
    lines.append("-" * 78)
    lines.append(
        f"  {'dialect':<14s} {'n_ref':>7s}  {'baseline':>10s} {'hybrid':>10s} {'delta':>9s}"
    )
    lines.append("-" * 78)
    for dialect in FOCUS_DIALECTS:
        b_tup = b_lidacc.get(dialect)
        h_tup = h_lidacc.get(dialect)
        b_acc = b_tup[2] if b_tup else None
        h_acc = h_tup[2] if h_tup else None
        n_show = (b_tup or h_tup or (0, 0, 0))[0]
        lines.append(
            f"  {dialect:<14s} {n_show:>7d}  "
            f"{_fmt_pct(b_acc):>10s} "
            f"{_fmt_pct(h_acc):>10s} "
            f"{_fmt_delta(b_acc, h_acc, invert=True):>9s}"
        )
    lines.append("")

    # ---- LID precision by pred_dialect（focus dialects） ----
    lines.append("[5] LID precision by pred_dialect (higher is better) — focus dialects")
    lines.append("-" * 78)
    lines.append(
        f"  {'dialect':<14s} {'pred_n(h)':>9s}  {'baseline':>10s} {'hybrid':>10s} {'delta':>9s}"
    )
    lines.append("-" * 78)
    for dialect in FOCUS_DIALECTS:
        b_p = b_lidprec.get(dialect, {}).get("precision")
        h_p = h_lidprec.get(dialect, {}).get("precision")
        h_pred_n = h_lidprec.get(dialect, {}).get("pred_n", 0)
        lines.append(
            f"  {dialect:<14s} {h_pred_n:>9d}  "
            f"{_fmt_pct(b_p):>10s} "
            f"{_fmt_pct(h_p):>10s} "
            f"{_fmt_delta(b_p, h_p, invert=True):>9s}"
        )
    lines.append("")

    lines.append("Legend:")
    lines.append("  CER 列  ↓ 表示更好（下降），↑ 表示变差（上升）；")
    lines.append("  LID 列  ↑ 表示更好（提升），↓ 表示变差（下降）；")
    lines.append("  ★ = focus dialects（本次路由改写覆盖的方言）")
    lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Baseline vs Hybrid summary builder")
    p.add_argument("--baseline_eval_dir", required=True, type=str)
    p.add_argument("--hybrid_eval_dir",   required=True, type=str)
    p.add_argument("--routed_jsonl",      required=True, type=str)
    p.add_argument("--run_tag",           default="p95",  type=str)
    p.add_argument("--tau_profile",       default="p95",  type=str)
    p.add_argument("--output",            required=True, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = build_summary(
        baseline_eval_dir=Path(args.baseline_eval_dir),
        hybrid_eval_dir=Path(args.hybrid_eval_dir),
        routed_jsonl=Path(args.routed_jsonl),
        run_tag=args.run_tag,
        tau_profile=args.tau_profile,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"[summarize] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

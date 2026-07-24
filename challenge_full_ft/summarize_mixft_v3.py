#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_mixft_v3.py
=====================

对比 outputs_vc_v2/checkpoint-500（现在最优基线）与 outputs_mixft_v3/best
在 challenge_full_ft/data/test.jsonl 上的评估结果，输出：
  - 整体 CER 对比
  - 每方言 CER Δ
  - LID 准确率 Δ

用法：
    python challenge_full_ft/summarize_mixft_v3.py

产出：
    challenge_full_ft/outputs_mixft_v3/vs_vc_v2.txt
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, Optional, Tuple


BASE_DIR = "/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft"
VC_V2_DIR = os.path.join(BASE_DIR, "outputs_vc_v2")
MIXFT_DIR = os.path.join(BASE_DIR, "outputs_mixft_v3")


ALL_ACCENTS = [
    "anhui", "cantonese", "changsha", "chaoshan", "dongbei", "henan",
    "kejia", "minnan", "nanchang", "nanjing", "shan1xi", "shan3xi",
    "shandong", "sichuan", "wuhan", "wuyu",
]


def read_overall_cer(wer_dir: str) -> Optional[float]:
    """从 wer_eval/result.wer 提取整体 CER 百分比。"""
    path = os.path.join(wer_dir, "result.wer")
    if not os.path.isfile(path):
        return None
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    for pat in [
        r"Overall[^\n]*?([0-9]+\.[0-9]+)\s*%",
        r"%WER\s+([0-9]+\.[0-9]+)",
        r"WER\s*=\s*([0-9]+\.[0-9]+)",
        r"CER\s*[:=]\s*([0-9]+\.[0-9]+)",
        r"([0-9]+\.[0-9]+)\s*%",
    ]:
        m = re.search(pat, txt)
        if m:
            return float(m.group(1))
    return None


def read_by_dialect(wer_dir: str) -> Dict[str, Tuple[int, float]]:
    """从 wer_eval/by_dialect_summary.txt 提取 {accent: (samples, cer%)}."""
    path = os.path.join(wer_dir, "by_dialect_summary.txt")
    if not os.path.isfile(path):
        return {}
    out: Dict[str, Tuple[int, float]] = {}
    for line in open(path, "r", encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("-") or line.lower().startswith("dialect"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            continue
        try:
            dia = parts[0]
            n = int(parts[1])
            wer = float(parts[-1].rstrip("%"))
            out[dia] = (n, wer)
        except (ValueError, IndexError):
            continue
    return out


def read_dialect_accuracy(wer_dir: str) -> Optional[float]:
    """从 wer_eval/dialect_accuracy.txt 提取 LID 准确率百分比。"""
    path = os.path.join(wer_dir, "dialect_accuracy.txt")
    if not os.path.isfile(path):
        return None
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    # 常见格式：Overall accuracy: xx.xx%
    for pat in [
        r"[Oo]verall\s+accuracy[^\n]*?([0-9]+\.[0-9]+)\s*%",
        r"[Aa]ccuracy[^\n]*?([0-9]+\.[0-9]+)\s*%",
        r"([0-9]+\.[0-9]+)\s*%",
    ]:
        m = re.search(pat, txt)
        if m:
            return float(m.group(1))
    return None


def read_dialect_accuracy_by_class(wer_dir: str) -> Dict[str, float]:
    """尝试解析每方言 LID 准确率（如果 dialect_accuracy.txt 提供）。"""
    path = os.path.join(wer_dir, "dialect_accuracy.txt")
    if not os.path.isfile(path):
        return {}
    out: Dict[str, float] = {}
    for line in open(path, "r", encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        # 期望形如：anhui: xx.xx% 或 anhui  350  xx.xx%
        m = re.match(r"^\s*([a-zA-Z0-9_]+)\s*[:\s]\s+.*?([0-9]+\.[0-9]+)\s*%\s*$", line)
        if m:
            dia = m.group(1)
            if dia in ALL_ACCENTS:
                out[dia] = float(m.group(2))
    return out


def find_best_ckpt(output_dir: str) -> str:
    p = os.path.join(output_dir, "best_ckpt.txt")
    if os.path.isfile(p):
        return open(p).read().strip()
    return "N/A"


def format_delta(new: Optional[float], base: Optional[float]) -> str:
    if new is None or base is None:
        return "  N/A "
    d = new - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


def main() -> None:
    # ---------- VC_v2 ---------- #
    vc_wer = os.path.join(VC_V2_DIR, "wer_eval")
    vc_overall = read_overall_cer(vc_wer)
    vc_by = read_by_dialect(vc_wer)
    vc_lid = read_dialect_accuracy(vc_wer)
    vc_lid_cls = read_dialect_accuracy_by_class(vc_wer)
    vc_ckpt = find_best_ckpt(VC_V2_DIR)

    # ---------- Mixft v3 ---------- #
    mix_wer = os.path.join(MIXFT_DIR, "wer_eval")
    mix_overall = read_overall_cer(mix_wer)
    mix_by = read_by_dialect(mix_wer)
    mix_lid = read_dialect_accuracy(mix_wer)
    mix_lid_cls = read_dialect_accuracy_by_class(mix_wer)
    mix_ckpt = find_best_ckpt(MIXFT_DIR)

    lines = []
    lines.append("======================================================================")
    lines.append(" MixFT v3 (外部 12 方言 + VC_v2 联合微调) vs VC_v2 checkpoint-500")
    lines.append("----------------------------------------------------------------------")
    lines.append(f" VC_v2 best ckpt   : {vc_ckpt}")
    lines.append(f" MixFT v3 best ckpt: {mix_ckpt}")
    lines.append("======================================================================")
    lines.append("")
    lines.append("[整体指标]")
    lines.append(f"  overall CER  VC_v2   = {vc_overall}")
    lines.append(f"  overall CER  MixFTv3 = {mix_overall}")
    lines.append(f"  Δ CER (mix - vc)     = {format_delta(mix_overall, vc_overall)}   (负值=更好)")
    lines.append("")
    lines.append(f"  LID acc      VC_v2   = {vc_lid}")
    lines.append(f"  LID acc      MixFTv3 = {mix_lid}")
    lines.append(f"  Δ LID acc            = {format_delta(mix_lid, vc_lid)}   (正值=更好)")
    lines.append("")
    lines.append("[按方言 CER (%)]")
    lines.append("  " + "  ".join([
        f"{'dialect':<10s}", f"{'n':>6s}",
        f"{'vc_v2':>8s}", f"{'mixft':>8s}", f"{'Δcer':>7s}",
        f"{'lid_vc':>7s}", f"{'lid_mix':>8s}", f"{'Δlid':>7s}",
    ]))
    lines.append("  " + "-" * 78)
    for acc in ALL_ACCENTS:
        vn, vc = vc_by.get(acc, (0, None))
        mn, mc = mix_by.get(acc, (0, None))
        n = vn or mn
        vc_s = f"{vc:.2f}" if vc is not None else "N/A"
        mc_s = f"{mc:.2f}" if mc is not None else "N/A"
        dc = format_delta(mc, vc)
        vlid = vc_lid_cls.get(acc)
        mlid = mix_lid_cls.get(acc)
        vlid_s = f"{vlid:.2f}" if vlid is not None else "N/A"
        mlid_s = f"{mlid:.2f}" if mlid is not None else "N/A"
        dlid = format_delta(mlid, vlid)
        lines.append("  " + "  ".join([
            f"{acc:<10s}", f"{n:>6d}",
            f"{vc_s:>8s}", f"{mc_s:>8s}", f"{dc:>7s}",
            f"{vlid_s:>7s}", f"{mlid_s:>8s}", f"{dlid:>7s}",
        ]))
    lines.append("======================================================================")

    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(MIXFT_DIR, "vs_vc_v2.txt")
    os.makedirs(MIXFT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[DONE] written -> {out_path}")


if __name__ == "__main__":
    main()

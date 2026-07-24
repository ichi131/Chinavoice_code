#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pick_best_ckpt.py
==================

从训练输出目录中挑出 eval 指标最好的 checkpoint 的绝对路径，并仅以单行
形式打印到 stdout，方便被 shell 通过 ``$(python pick_best_ckpt.py ...)`` 捕获。

挑选顺序（优先级从高到低）：
1. ``<output_dir>/best_ckpt.txt``（由 ``qwen3_asr_sft_full.py`` 训练收尾时写入）
2. ``<output_dir>/trainer_state.json`` 中的 ``best_model_checkpoint``
3. 扫描所有 ``checkpoint-*/trainer_state.json``，按 ``eval_<metric>`` 挑最优

所有调试信息都走 stderr，避免污染 stdout。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _read_best_ckpt_txt(output_dir: str) -> Optional[str]:
    p = os.path.join(output_dir, "best_ckpt.txt")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            path = f.read().strip()
        if path and os.path.isdir(path):
            return path
    except OSError as e:
        _log(f"[pick_best_ckpt] cannot read {p}: {e}")
    return None


def _read_root_trainer_state(output_dir: str) -> Optional[str]:
    p = os.path.join(output_dir, "trainer_state.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _log(f"[pick_best_ckpt] cannot parse {p}: {e}")
        return None
    best = state.get("best_model_checkpoint")
    if isinstance(best, str) and best and os.path.isdir(best):
        return best
    return None


def _iter_ckpt_dirs(output_dir: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    if not os.path.isdir(output_dir):
        return out
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            out.append((int(m.group(1)), path))
    out.sort(key=lambda x: x[0])
    return out


def _extract_metric(trainer_state_path: str, metric_key: str) -> Optional[float]:
    try:
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _log(f"[pick_best_ckpt] cannot parse {trainer_state_path}: {e}")
        return None

    key = f"eval_{metric_key}" if not metric_key.startswith("eval_") else metric_key
    last_val: Optional[float] = None
    for entry in state.get("log_history", []):
        if key in entry:
            try:
                last_val = float(entry[key])
            except (TypeError, ValueError):
                continue
    return last_val


def pick_best(
    output_dir: str,
    metric: str,
    greater_is_better: bool,
    fallback_best_dir: Optional[str] = None,
) -> Optional[str]:
    p = _read_best_ckpt_txt(output_dir)
    if p:
        _log("[pick_best_ckpt] source = best_ckpt.txt")
        return p

    p = _read_root_trainer_state(output_dir)
    if p:
        _log("[pick_best_ckpt] source = trainer_state.json@root")
        return p

    ckpts = _iter_ckpt_dirs(output_dir)
    if not ckpts:
        _log(f"[pick_best_ckpt] no checkpoint-* found under {output_dir}")
        if fallback_best_dir and os.path.isdir(fallback_best_dir):
            _log(f"[pick_best_ckpt] fallback_best_dir = {fallback_best_dir}")
            return os.path.abspath(fallback_best_dir)
        return None

    best_path: Optional[str] = None
    best_val: Optional[float] = None
    for step, path in ckpts:
        ts = os.path.join(path, "trainer_state.json")
        val = _extract_metric(ts, metric)
        if val is None:
            _log(f"[pick_best_ckpt] step={step} MISS metric={metric}")
            continue
        _log(f"[pick_best_ckpt] step={step} eval_{metric}={val} path={path}")
        if best_val is None:
            best_val, best_path = val, path
        else:
            better = (val > best_val) if greater_is_better else (val < best_val)
            if better:
                best_val, best_path = val, path

    if best_path is None:
        _log("[pick_best_ckpt] no ckpt contains eval_%s; fallback to latest" % metric)
        best_path = ckpts[-1][1]

    return best_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pick best checkpoint dir from a HF Trainer output_dir."
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--metric", type=str, default="eval_loss")
    p.add_argument("--greater_is_better", type=int, default=0)
    p.add_argument("--fallback_best_dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    fallback = os.path.abspath(args.fallback_best_dir) if args.fallback_best_dir else None

    metric = args.metric[len("eval_"):] if args.metric.startswith("eval_") else args.metric
    greater = bool(args.greater_is_better)

    best = pick_best(
        output_dir=output_dir,
        metric=metric,
        greater_is_better=greater,
        fallback_best_dir=fallback,
    )
    if not best:
        _log("[pick_best_ckpt] FAILED: no candidate ckpt found.")
        sys.exit(1)

    print(os.path.abspath(best))


if __name__ == "__main__":
    main()

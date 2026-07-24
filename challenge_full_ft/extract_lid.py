#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_lid.py
==============

从 infer_evalset 产出的 result.jsonl 中抽取语种/方言识别（LID）字段，
写成比赛要求的 lid.jsonl 格式：

每行：{"key": "eval_000001", "dialect": "chaoshan"}

用法：
    python extract_lid.py --pred /abs/result.jsonl --out /abs/lid.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="infer_evalset 产出的 result.jsonl")
    p.add_argument("--out", required=True, help="输出 lid.jsonl 路径")
    p.add_argument(
        "--fallback", default="",
        help="pred_dialect 为空时的兜底方言（默认留空字符串）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.pred):
        raise SystemExit(f"pred jsonl not found: {args.pred}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    n_total = n_empty = 0
    dialect_cnt: Counter = Counter()

    with open(args.pred, "r", encoding="utf-8") as fin, \
         open(args.out,  "w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                o = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[extract_lid][WARN] bad json line {line_no}: {e}", file=sys.stderr)
                continue

            key     = o.get("utt_id") or o.get("key") or ""
            dialect = (o.get("pred_dialect") or "").strip()
            if not dialect:
                n_empty += 1
                dialect = args.fallback

            fout.write(
                json.dumps(
                    {"key": key, "dialect": dialect},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            n_total += 1
            if dialect:
                dialect_cnt[dialect] += 1

    print(f"[extract_lid] pred        = {args.pred}")
    print(f"[extract_lid] out         = {args.out}")
    print(f"[extract_lid] total       = {n_total}")
    print(f"[extract_lid] empty_pred  = {n_empty}  (fallback={args.fallback!r})")
    print("[extract_lid] dialect distribution:")
    for d, c in dialect_cnt.most_common():
        print(f"    {d:<15s} {c}")


if __name__ == "__main__":
    main()

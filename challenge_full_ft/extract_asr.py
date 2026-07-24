#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_asr.py
==============

从 infer_evalset 产出的 result.jsonl 中抽取 ASR 文本字段，
写成比赛要求的 asr.jsonl 格式：

每行：{"key": "eval_000001", "text": "我现在可以出去的了"}

用法：
    python extract_asr.py --pred /abs/result.jsonl --out /abs/asr.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="infer_evalset 产出的 result.jsonl")
    p.add_argument("--out", required=True, help="输出 asr.jsonl 路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.pred):
        raise SystemExit(f"pred jsonl not found: {args.pred}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    n_total = n_empty = n_err = 0
    with open(args.pred, "r", encoding="utf-8") as fin, \
         open(args.out,  "w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                o = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[extract_asr][WARN] bad json line {line_no}: {e}", file=sys.stderr)
                continue

            key  = o.get("utt_id") or o.get("key") or ""
            text = (o.get("pred_text") or "").strip()
            err  = (o.get("error") or "").strip()

            if err:
                n_err += 1
            if not text:
                n_empty += 1

            fout.write(
                json.dumps(
                    {"key": key, "text": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            n_total += 1

    print(f"[extract_asr] pred        = {args.pred}")
    print(f"[extract_asr] out         = {args.out}")
    print(f"[extract_asr] total       = {n_total}")
    print(f"[extract_asr] empty_text  = {n_empty}")
    print(f"[extract_asr] error_rows  = {n_err}")


if __name__ == "__main__":
    main()

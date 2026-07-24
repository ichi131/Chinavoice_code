#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
postprocess_pred_evalset.py
===========================

对 infer_test.py 在**无参考评估集**上的输出 JSONL 做后处理，产出两类文件：

1. ``result.jsonl``（每行一条）：
       {"utt_id":       "eval_000001",
        "audio_path":   "/abs/.../eval_000001.wav",
        "pred_text":    "识别文本",
        "pred_dialect": "chaoshan",
        "pred_full":    "language Chinese chaoshan<asr_text>识别文本",
        "error":        ""}

2. ``result.trn``（Kaldi 风格：``<pred_text> (<utt_id>)``），
   方便直接拿去做后续 CER 计算 / 提交。

用法：
    python postprocess_pred_evalset.py \\
        --pred /abs/pred_test.jsonl \\
        --out_dir /abs/infer_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="infer_test.py 产出的合并 jsonl")
    p.add_argument("--out_dir", required=True, help="后处理产物保存目录")
    p.add_argument(
        "--result_name", default="result.jsonl",
        help="精简 jsonl 文件名（默认 result.jsonl）",
    )
    p.add_argument(
        "--trn_name", default="result.trn",
        help="Kaldi 风格文本文件名（默认 result.trn）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.pred):
        raise SystemExit(f"pred jsonl not found: {args.pred}")
    os.makedirs(args.out_dir, exist_ok=True)

    result_path = os.path.join(args.out_dir, args.result_name)
    trn_path    = os.path.join(args.out_dir, args.trn_name)

    n_total = n_err = n_empty = 0
    dialect_cnt: Counter = Counter()

    with open(args.pred, "r", encoding="utf-8") as fin, \
         open(result_path, "w", encoding="utf-8") as f_res, \
         open(trn_path,    "w", encoding="utf-8") as f_trn:
        for line_no, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                o = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[postprocess][WARN] bad json {args.pred}:{line_no}: {e}", file=sys.stderr)
                continue

            utt_id       = o.get("utt_id", "")
            audio_path   = o.get("audio_path", "")
            pred_text    = (o.get("pred_text") or "").strip()
            pred_dialect = (o.get("pred_dialect") or "").strip()
            pred_full    = (o.get("pred_full") or "").strip()
            err          = (o.get("error") or "").strip()

            n_total += 1
            if err:
                n_err += 1
            if not pred_text:
                n_empty += 1
            if pred_dialect:
                dialect_cnt[pred_dialect] += 1

            f_res.write(json.dumps({
                "utt_id":       utt_id,
                "audio_path":   audio_path,
                "pred_text":    pred_text,
                "pred_dialect": pred_dialect,
                "pred_full":    pred_full,
                "error":        err,
            }, ensure_ascii=False) + "\n")

            # Kaldi 风格：一行一条 "text (utt_id)"
            f_trn.write(f"{pred_text} ({utt_id})\n")

    print(f"[postprocess] pred        = {args.pred}")
    print(f"[postprocess] result.jsonl= {result_path}")
    print(f"[postprocess] result.trn  = {trn_path}")
    print(f"[postprocess] total       = {n_total}")
    print(f"[postprocess] error       = {n_err}")
    print(f"[postprocess] empty_pred  = {n_empty}")
    if dialect_cnt:
        print("[postprocess] pred_dialect distribution (top 20):")
        for d, c in dialect_cnt.most_common(20):
            print(f"    {d:<15s} {c}")


if __name__ == "__main__":
    main()

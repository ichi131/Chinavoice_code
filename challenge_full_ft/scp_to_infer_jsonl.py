#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scp_to_infer_jsonl.py
=====================

把比赛评估集的 ``wav.scp`` 转成 ``infer_test.py`` 认可的 SFT 格式 JSONL。

wav.scp 每行两列 ``<key> <path>``，``<path>`` 可为**相对路径**（相对于 ``wav.scp``
所在目录的父目录）或**绝对路径**；本脚本会自动补齐为绝对路径。

评估集没有参考文本，所以输出的 ``text``/``accent`` 均置空，仅供无参考推理使用。

输出 JSONL 每行：
    {"audio": "/abs/.../eval_000001.wav",
     "text":  "",
     "prompt":"",
     "key":   "eval_000001",
     "accent":""}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def resolve_wav_path(raw_path: str, base_dir: str) -> str:
    """把 wav.scp 里可能是相对路径的字符串解析成绝对路径。"""
    if os.path.isabs(raw_path):
        return raw_path
    # 常见约定：路径相对 wav.scp 所在目录的**父目录**（如 ``evaluation_set/wav/xxx.wav``）
    cand = os.path.normpath(os.path.join(base_dir, raw_path))
    if os.path.exists(cand):
        return cand
    # 兜底：直接相对 wav.scp 目录
    cand2 = os.path.normpath(os.path.join(os.path.dirname(base_dir + os.sep), raw_path))
    return cand if not os.path.exists(cand2) else cand2


def convert(scp_path: str, out_jsonl: str, check_audio: bool) -> int:
    if not os.path.isfile(scp_path):
        raise FileNotFoundError(f"wav.scp 不存在: {scp_path}")

    # wav.scp 相对路径的基准目录：wav.scp 所在目录的父目录
    # 例如 wav.scp 在 .../chinavoices_challenge/evaluation_set/ ，
    # 内部路径写作 evaluation_set/wav/xxx.wav ，则 base = .../chinavoices_challenge/
    scp_dir = os.path.dirname(os.path.abspath(scp_path))
    base_dir = os.path.dirname(scp_dir)

    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)

    n_ok = n_missing = n_bad = 0
    with open(scp_path, "r", encoding="utf-8") as fin, \
         open(out_jsonl, "w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(None, 1)
            if len(parts) != 2:
                n_bad += 1
                print(f"[scp_to_jsonl][WARN] {scp_path}:{line_no} format bad: {raw!r}", file=sys.stderr)
                continue
            key, wav_rel = parts[0], parts[1]
            wav_abs = resolve_wav_path(wav_rel, base_dir)

            if check_audio and not os.path.isfile(wav_abs):
                n_missing += 1
                print(f"[scp_to_jsonl][WARN] wav not found (key={key}): {wav_abs}", file=sys.stderr)
                continue

            rec = {
                "audio":  wav_abs,
                "text":   "",     # 无参考文本
                "prompt": "",
                "key":    key,
                "accent": "",     # 无参考方言
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[scp_to_jsonl] scp     = {scp_path}")
    print(f"[scp_to_jsonl] base    = {base_dir}")
    print(f"[scp_to_jsonl] out     = {out_jsonl}")
    print(f"[scp_to_jsonl] ok={n_ok}  missing={n_missing}  bad={n_bad}")
    return n_ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert wav.scp -> infer_test.py compatible jsonl")
    p.add_argument("--scp", type=str, required=True, help="wav.scp 绝对路径")
    p.add_argument("--out", type=str, required=True, help="输出 jsonl 路径")
    p.add_argument(
        "--check_audio", type=int, default=0,
        help="1=逐条 stat wav 是否存在（10w+ 条时会较慢），0=跳过检查（默认）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = convert(args.scp, args.out, bool(args.check_audio))
    if n == 0:
        raise SystemExit("[scp_to_jsonl] no valid line converted")


if __name__ == "__main__":
    main()

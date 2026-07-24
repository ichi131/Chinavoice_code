#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_data.py
================

把 `challenge_data_speaker` 目录下的三份 jsonl（train/val/test）转换为
`Qwen3-ASR/finetuning/qwen3_asr_sft.py` 可以直接读取的 SFT JSONL 格式，
并保留 `key / accent` 字段，方便后续推理与 CER 评估。

输入样例（每行一个 json）：
    {"key": "chaoshan_000459",
     "wav_path": "/.../chaoshan/wav/chaoshan_000459.wav",
     "text": "嗯本钱都用了一万了",
     "accent": "chaoshan"}

输出样例（每行一个 json）：
    {"audio":  "/.../chaoshan/wav/chaoshan_000459.wav",
     "text":   "language Chinese chaoshan<asr_text>嗯本钱都用了一万了",
     "prompt": "",
     "key":    "chaoshan_000459",
     "accent": "chaoshan"}

用法：
    python challenge_full_ft/prepare_data.py \
        --src_dir /mnt/wfs/.../challenge_data_speaker/ \
        --out_dir /mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data \
        --check_audio_exists 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


DEFAULT_SRC_DIR = (
    "/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/"
    "user_ichiwang/data/challenge_data_speaker"
)
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)


def build_target_text(accent: str, text: str) -> str:
    """构造与 ChinaVoices baseline 对齐的 assistant 目标文本：
    ``language Chinese <accent><asr_text><原始 text>``
    """
    accent = (accent or "").strip()
    text = (text or "").strip()
    return f"language Chinese {accent}<asr_text>{text}"


def convert_split(
    src_path: str,
    out_path: str,
    check_audio_exists: bool = False,
) -> Tuple[int, int, Counter]:
    """把单个 split 的 jsonl 转换为 SFT 格式。

    Returns
    -------
    (n_total, n_missing_audio, accent_counter)
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"src jsonl not found: {src_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_total = 0
    n_missing = 0
    accent_counter: Counter = Counter()

    with open(src_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(
                    f"[WARN] skip invalid json at {src_path}:{line_no}: {e}",
                    file=sys.stderr,
                )
                continue

            key = obj.get("key", "")
            wav_path = obj.get("wav_path", "")
            text = obj.get("text", "")
            accent = obj.get("accent", "")

            if check_audio_exists and wav_path and not os.path.isfile(wav_path):
                n_missing += 1
                # 缺失音频则跳过该条，但不中断
                continue

            new_obj: Dict[str, object] = {
                "audio": wav_path,
                "text": build_target_text(accent, text),
                "prompt": "",
                "key": key,
                "accent": accent,
            }
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            n_total += 1
            accent_counter[accent] += 1

    return n_total, n_missing, accent_counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert challenge_data_speaker jsonl → Qwen3-ASR SFT jsonl."
    )
    p.add_argument("--src_dir", type=str, default=DEFAULT_SRC_DIR,
                   help="数据根目录（包含 data_train.jsonl / data_val.jsonl / data_test.jsonl）")
    p.add_argument("--train_name", type=str, default="data_train.jsonl")
    p.add_argument("--val_name",   type=str, default="data_val.jsonl")
    p.add_argument("--test_name",  type=str, default="data_test.jsonl")
    p.add_argument("--out_dir",    type=str, default=DEFAULT_OUT_DIR,
                   help="输出目录（默认 challenge_full_ft/data/）")
    p.add_argument("--check_audio_exists", type=int, default=0,
                   help="是否检查 wav 文件是否存在，缺失则跳过（不中断）；0=否，1=是")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = os.path.abspath(args.src_dir)
    out_dir = os.path.abspath(args.out_dir)
    check_audio = bool(args.check_audio_exists)

    os.makedirs(out_dir, exist_ok=True)

    splits = [
        ("train", args.train_name, "train.jsonl"),
        ("val",   args.val_name,   "val.jsonl"),
        ("test",  args.test_name,  "test.jsonl"),
    ]

    print(f"[prepare_data] src_dir = {src_dir}")
    print(f"[prepare_data] out_dir = {out_dir}")
    print(f"[prepare_data] check_audio_exists = {check_audio}")

    for split, src_name, out_name in splits:
        src_path = os.path.join(src_dir, src_name)
        out_path = os.path.join(out_dir, out_name)

        print(f"\n[prepare_data] === split={split} ===")
        print(f"[prepare_data]   src = {src_path}")
        print(f"[prepare_data]   out = {out_path}")

        n_total, n_missing, accent_counter = convert_split(
            src_path=src_path,
            out_path=out_path,
            check_audio_exists=check_audio,
        )

        print(f"[prepare_data]   total kept = {n_total}")
        if check_audio:
            print(f"[prepare_data]   missing wav (skipped) = {n_missing}")
        # 按 accent 输出样本分布，按数量降序
        print(f"[prepare_data]   by accent ({len(accent_counter)} unique):")
        for accent, cnt in sorted(
            accent_counter.items(), key=lambda x: (-x[1], x[0])
        ):
            print(f"[prepare_data]     - {accent:<16s} {cnt}")

    print("\n[prepare_data] done.")


if __name__ == "__main__":
    main()

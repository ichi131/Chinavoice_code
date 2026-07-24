#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 data_mixft_v5 数据集转换为 FireRedASR finetune 脚本兼容的 VC 字段格式。

输入字段（mix_v5）：
    {"audio": <path>, "text": "language Chinese<asr_text><真实转写>",
     "prompt": "", "key": "...", "accent": "...", "src": "..."}

输出字段（VC 格式，与 VC_data_v2/data_train_vc.jsonl 一致）：
    {"key": "...", "wav_path": <path>, "text": "<真实转写>", "accent": "..."}

用法：
    python3 convert_mixft_v5_to_vc_format.py \
        --input  /path/to/train.jsonl \
        --output /path/to/data_train_vc_mix.jsonl

设计要点：
  - 只做字段重命名 + 前缀剥离；不改动语义
  - 若 text 中不含 "<asr_text>" 标记，则整条丢弃并计入 skipped
  - 若字段缺失（audio/text/key/accent 任一为空），丢弃并计入 skipped
  - 只读源文件，不覆盖任何已有输入
"""

import argparse
import json
import os
import sys

ASR_MARKER = "<asr_text>"


def strip_prefix(text: str) -> str:
    """剥掉 'language Chinese<asr_text>' 前缀，只保留真实转写。"""
    idx = text.find(ASR_MARKER)
    if idx < 0:
        return ""
    return text[idx + len(ASR_MARKER):].strip()


def convert(input_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n_in = n_out = n_skip = 0
    src_stats = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_skip += 1
                continue

            key = d.get("key", "")
            wav_path = d.get("audio") or d.get("wav_path", "")
            raw_text = d.get("text", "")
            accent = d.get("accent", "")
            src = d.get("src", "unknown")

            if not (key and wav_path and raw_text and accent):
                n_skip += 1
                continue

            pure_text = strip_prefix(raw_text) if ASR_MARKER in raw_text else raw_text.strip()
            if not pure_text:
                n_skip += 1
                continue

            out_obj = {
                "key": key,
                "wav_path": wav_path,
                "text": pure_text,
                "accent": accent,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_out += 1
            src_stats[src] = src_stats.get(src, 0) + 1

    print(f"[convert] input={input_path}")
    print(f"[convert] output={output_path}")
    print(f"[convert] in={n_in} out={n_out} skipped={n_skip}")
    if src_stats:
        print(f"[convert] by src:")
        for k in sorted(src_stats, key=lambda x: -src_stats[x]):
            print(f"    {k:24s} {src_stats[k]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="mix_v5 jsonl 输入")
    ap.add_argument("--output", required=True, help="VC 格式 jsonl 输出")
    args = ap.parse_args()
    convert(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

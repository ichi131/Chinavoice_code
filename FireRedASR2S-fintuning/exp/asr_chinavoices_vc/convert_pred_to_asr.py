#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 decode_asr_chinavoices.py 的推理产物 pred_evaluation.jsonl
转换为竞赛提交格式 asr.jsonl。

转换规则：
1. 只保留 utt_id -> key、pred_text -> text 两个字段
2. 过滤掉 text 中的特殊 token（<sos>/<eos>/<unk>/<pad>/<blank>/<asr_text>/<mask> 等
   形如 <xxx> 的 ASR 特殊标记），以及首尾空白
3. 按 eval_XXXXXX 后缀数字升序排序，确保和 evaluation.jsonl 逐行对齐
4. 使用 separators=(",", ":") 紧凑序列化，冒号/逗号后无空格
"""

import json
import re
from pathlib import Path

SRC = "/mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning/exp/asr_chinavoices_vc/pred_evaluation.jsonl"
DST = "/mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning/exp/asr_chinavoices_vc/asr.jsonl"

# 需要清理的 ASR 特殊 token：形如 <xxx>，覆盖 sos/eos/unk/pad/blank/asr_text/mask 等
SPECIAL_TOKEN_RE = re.compile(r"<[^<>\s]{1,32}>")


def clean_text(text: str) -> str:
    if not text:
        return ""
    # 去除所有 <xxx> 特殊标记
    text = SPECIAL_TOKEN_RE.sub("", text)
    # 折叠因删除标记留下的多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sort_key(row):
    k = row["key"]
    try:
        return (0, int(k.split("_")[-1]), k)
    except Exception:
        return (1, 0, k)


def main():
    src_path = Path(SRC)
    dst_path = Path(DST)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_lines = 0
    n_empty_after_clean = 0
    n_had_special = 0

    with open(src_path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            obj = json.loads(line)
            raw = obj.get("pred_text", "") or ""
            cleaned = clean_text(raw)
            if cleaned != raw.strip():
                n_had_special += 1
            if not cleaned:
                n_empty_after_clean += 1
            rows.append({"key": obj["utt_id"], "text": cleaned})

    rows.sort(key=sort_key)

    with open(dst_path, "w", encoding="utf-8") as fout:
        for r in rows:
            fout.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"source              = {src_path}")
    print(f"output              = {dst_path}")
    print(f"total_lines         = {len(rows)}")
    print(f"lines_had_special   = {n_had_special}")
    print(f"empty_after_clean   = {n_empty_after_clean}")


if __name__ == "__main__":
    main()

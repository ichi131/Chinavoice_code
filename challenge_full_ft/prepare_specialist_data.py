#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_specialist_data.py
==========================

16 方言"单语种专用 ASR"微调的数据处理脚本。参考 prepare_wuyu_data.py 结构，
将 accent 参数化，产出：
  - {out_dir}/{accent}/train.jsonl
  - {out_dir}/{accent}/val.jsonl
  - {out_dir}/{accent}/test.jsonl

统一 SFT 格式：
    {"audio": ..., "text": "language Chinese {accent}<asr_text>{原文}",
     "prompt": "", "key": ..., "accent": "{accent}"}

去重策略：
  - 所有音色增广样本（key 含 __to__ / __aug_ 后缀）全部保留
  - 原始样本按去后缀 base key 去重，同一 base key 只保留一份
  - challenge_raw 版本优先覆盖 VC_data_v2 版本
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 通用常量
# ---------------------------------------------------------------------------

# 挑战集官方 16 方言列表
ALL_ACCENTS = [
    "anhui", "cantonese", "changsha", "chaoshan", "dongbei", "henan",
    "kejia", "minnan", "nanchang", "nanjing", "shan1xi", "shan3xi",
    "shandong", "sichuan", "wuhan", "wuyu",
]

AUG_SUFFIX_RE = re.compile(r"(?:__to__|__aug_).*$")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def iter_jsonl(path: str) -> Iterable[Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"jsonl not found: {path}")
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[WARN] skip invalid json at {path}:{line_no}: {e}",
                      file=sys.stderr)


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fout:
        for obj in rows:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def target_prefix(accent: str) -> str:
    return f"language Chinese {accent}<asr_text>"


def build_target(text: str, accent: str) -> str:
    text = (text or "").strip()
    return f"{target_prefix(accent)}{text}"


def strip_aug_suffix(key: str) -> str:
    if not key:
        return ""
    return AUG_SUFFIX_RE.sub("", key)


def is_augmented_key(key: str) -> str:
    if "__to__" in key:
        return "__to__"
    if "__aug_" in key:
        return "__aug_"
    return ""


def is_target_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# 单行规范化
# ---------------------------------------------------------------------------

def normalize_row_vc_v2(obj: Dict, accent: str, check_audio: bool
                        ) -> Tuple[Optional[Dict], str]:
    """VC_data_v2 格式：{key, wav_path, text, accent}"""
    acc = (obj.get("accent") or "").strip()
    if acc != accent:
        return None, "accent_mismatch"
    audio = (obj.get("wav_path") or "").strip()
    text = (obj.get("text") or "").strip()
    key = (obj.get("key") or "").strip()
    if not audio:
        return None, "empty_audio"
    if not text:
        return None, "empty_text"
    if check_audio and not os.path.isfile(audio):
        return None, "missing_audio"
    if not key:
        key = os.path.splitext(os.path.basename(audio))[0]
    return {
        "audio": audio,
        "text": build_target(text, accent),
        "prompt": "",
        "key": key,
        "accent": accent,
    }, ""


def normalize_row_challenge_raw(obj: Dict, accent: str, check_audio: bool
                                ) -> Tuple[Optional[Dict], str]:
    """
    challenge_full_ft/data 里已经是 SFT 格式：
        {audio, text="language Chinese {accent}<asr_text>{原文}", prompt, key, accent}
    这里把 target 前缀强制改成当前 accent（虽然本来就应该匹配）。
    """
    acc = (obj.get("accent") or "").strip()
    if acc != accent:
        return None, "accent_mismatch"
    audio = (obj.get("audio") or "").strip()
    text = (obj.get("text") or "").strip()
    key = (obj.get("key") or "").strip()
    if not audio:
        return None, "empty_audio"
    if not text:
        return None, "empty_text"
    if check_audio and not os.path.isfile(audio):
        return None, "missing_audio"

    if not text.startswith(target_prefix(accent)):
        m = re.match(r"^language Chinese \S+<asr_text>(.*)$", text, re.DOTALL)
        raw_text = m.group(1) if m else text
        text = build_target(raw_text, accent)

    if not key:
        key = os.path.splitext(os.path.basename(audio))[0]
    return {
        "audio": audio,
        "text": text,
        "prompt": "",
        "key": key,
        "accent": accent,
    }, ""


def _load_wuyu_style(path: str, accent: str, source_tag: str,
                     check_audio: bool) -> List[Dict]:
    if source_tag == "vc_v2":
        norm = normalize_row_vc_v2
    elif source_tag == "challenge_raw":
        norm = normalize_row_challenge_raw
    else:
        raise ValueError(f"unknown source_tag: {source_tag}")

    rows: List[Dict] = []
    for obj in iter_jsonl(path):
        new_obj, _reason = norm(obj, accent, check_audio)
        if new_obj is not None:
            rows.append(new_obj)
    return rows


# ---------------------------------------------------------------------------
# 合并去重
# ---------------------------------------------------------------------------

def _dedup_and_merge(vc_rows: List[Dict], raw_rows: List[Dict], tag: str
                     ) -> List[Dict]:
    aug_rows: List[Dict] = []
    orig_by_key: Dict[str, Dict] = {}
    stats = Counter()

    # 先处理 VC 侧
    for r in vc_rows:
        aug_tag = is_augmented_key(r["key"])
        if aug_tag:
            aug_rows.append(r)
            stats[aug_tag] += 1
        else:
            base_key = strip_aug_suffix(r["key"])
            if base_key not in orig_by_key:
                orig_by_key[base_key] = r
                orig_by_key[base_key]["__from"] = "vc"
                stats["orig_vc"] += 1

    # 再处理 challenge_raw 侧（同 key 覆盖）
    for r in raw_rows:
        aug_tag = is_augmented_key(r["key"])
        if aug_tag:
            aug_rows.append(r)
            stats[aug_tag] += 1
        else:
            base_key = strip_aug_suffix(r["key"])
            if base_key in orig_by_key and \
                    orig_by_key[base_key].get("__from") == "vc":
                stats["orig_vc"] -= 1
            r["__from"] = "raw"
            orig_by_key[base_key] = r
            stats["orig_raw"] += 1

    orig_rows = list(orig_by_key.values())
    for r in orig_rows:
        r.pop("__from", None)

    merged = aug_rows + orig_rows
    print(f"[merge:{tag}] __to__ aug          = {stats['__to__']}")
    print(f"[merge:{tag}] __aug_ aug          = {stats['__aug_']}")
    print(f"[merge:{tag}] orig from vc_v2     = {stats['orig_vc']}")
    print(f"[merge:{tag}] orig from raw       = {stats['orig_raw']}")
    print(f"[merge:{tag}] merged total        = {len(merged)}")
    return merged


# ---------------------------------------------------------------------------
# 主命令
# ---------------------------------------------------------------------------

def build_specialist(args: argparse.Namespace) -> None:
    accent = args.accent
    if accent not in ALL_ACCENTS:
        print(f"[build] ERROR: --accent {accent} not in {ALL_ACCENTS}",
              file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.abspath(os.path.join(args.out_root, accent))
    os.makedirs(out_dir, exist_ok=True)

    train_out = os.path.join(out_dir, "train.jsonl")
    val_out = os.path.join(out_dir, "val.jsonl")
    test_out = os.path.join(out_dir, "test.jsonl")

    force = bool(args.force) or os.environ.get("FORCE_REBUILD", "0") == "1"

    if not force and is_target_nonempty(train_out) \
            and is_target_nonempty(val_out) \
            and is_target_nonempty(test_out):
        print(f"[build:{accent}] all target files exist and non-empty, SKIP.")
        return

    check_audio = bool(args.check_audio_exists)

    print(f"[build:{accent}] === Train ===")
    vc_tr = _load_wuyu_style(args.vc_train, accent, "vc_v2", check_audio)
    raw_tr = _load_wuyu_style(args.raw_train, accent, "challenge_raw",
                              check_audio)
    print(f"[build:{accent}] vc_v2   train rows  = {len(vc_tr)}")
    print(f"[build:{accent}] raw     train rows  = {len(raw_tr)}")
    merged_tr = _dedup_and_merge(vc_tr, raw_tr, f"{accent}:train")

    print(f"\n[build:{accent}] === Val ===")
    vc_va = _load_wuyu_style(args.vc_val, accent, "vc_v2", check_audio)
    raw_va = _load_wuyu_style(args.raw_val, accent, "challenge_raw",
                              check_audio)
    print(f"[build:{accent}] vc_v2   val rows    = {len(vc_va)}")
    print(f"[build:{accent}] raw     val rows    = {len(raw_va)}")
    merged_va = _dedup_and_merge(vc_va, raw_va, f"{accent}:val")

    print(f"\n[build:{accent}] === Test (从官方 test.jsonl 抽取) ===")
    n_in = n_out = 0
    with open(test_out, "w", encoding="utf-8") as fout:
        for obj in iter_jsonl(args.raw_test):
            n_in += 1
            if (obj.get("accent") or "").strip() != accent:
                continue
            # test 保持原样（不改 text 前缀，以对齐官方评估）
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[build:{accent}] test scanned        = {n_in}")
    print(f"[build:{accent}] test kept ({accent}) = {n_out}")
    if n_out == 0:
        print(f"[build:{accent}] ERROR: no {accent} row in test set!",
              file=sys.stderr)
        sys.exit(1)

    if not merged_tr or not merged_va:
        print(f"[build:{accent}] ERROR: train or val empty!", file=sys.stderr)
        sys.exit(1)

    n_tr = write_jsonl(train_out, merged_tr)
    n_va = write_jsonl(val_out, merged_va)

    print(f"\n[build:{accent}] written train       = {n_tr}  ({train_out})")
    print(f"[build:{accent}] written val         = {n_va}  ({val_out})")
    print(f"[build:{accent}] written test        = {n_out}  ({test_out})")
    print(f"[build:{accent}] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="16 方言单语种专用 ASR 微调数据处理脚本。")
    p.add_argument("--accent", required=True,
                   help=f"目标方言，需在 {ALL_ACCENTS} 之中")
    p.add_argument("--vc_train", required=True,
                   help="VC_data_v2/data_train_vc.jsonl 路径")
    p.add_argument("--vc_val", required=True,
                   help="VC_data_v2/data_val_vc.jsonl 路径")
    p.add_argument("--raw_train", required=True,
                   help="challenge_full_ft/data/train.jsonl (SFT 格式)")
    p.add_argument("--raw_val", required=True,
                   help="challenge_full_ft/data/val.jsonl (SFT 格式)")
    p.add_argument("--raw_test", required=True,
                   help="challenge_full_ft/data/test.jsonl (SFT 格式)")
    p.add_argument("--out_root", required=True,
                   help="输出根目录，产物写到 {out_root}/{accent}/")
    p.add_argument("--check_audio_exists", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="强制重跑（也可 FORCE_REBUILD=1）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    build_specialist(args)


if __name__ == "__main__":
    main()

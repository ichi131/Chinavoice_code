#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_mixft_v4_data.py
========================

混合方言联合微调 v4 数据准备脚本（温度采样版）。

相比 v3 的核心变化：
    v3：外部数据每方言硬截断到 25k + VC_v2 全并 → 最少 kejia 5.5k / 最多 nanjing 36k
    v4：合并所有可用数据（外部+VC_v2）后，对每个 accent 做“温度采样”
        weight_i ∝ (n_i) ** alpha
        target_i = round( total_budget * weight_i / sum(weight) )
        然后再上采样/下采样每个 accent 到 target_i

    alpha=1.0   → 完全按比例（等价于不采样，与 v3 类似）
    alpha=0.5   → 温度采样（推荐首选，压平但不完全均衡）
    alpha=0.0   → 完全均衡（每个 accent 相同数量，激进）

输出：
    - challenge_full_ft/data_mixft_v4/train.jsonl
    - challenge_full_ft/data_mixft_v4/val.jsonl
    - challenge_full_ft/data_mixft_v4/test.jsonl
    - challenge_full_ft/data_mixft_v4/stats.txt

用法：
    python challenge_full_ft/prepare_mixft_v4_data.py \
        --out_dir challenge_full_ft/data_mixft_v4 \
        --per_accent_max 25000 \
        --alpha 0.5 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# --------------------------- 配置区 --------------------------- #

EXT_ROOT = "/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/challenge-data"

EXT_SOURCES: List[Tuple[str, str]] = [
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/hefei_anhui.jsonl",                                       "kespeech_anhui"),
    (f"{EXT_ROOT}/wenetspeech-yue/data_50h/meta_50h_converted.jsonl",                              "wenetspeech_yue"),
    (f"{EXT_ROOT}/Changsha_Dialect_Conversational_Speech_Corpus/changsha_segments_manifest.jsonl", "corpus_changsha"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/dongbei.jsonl",                                           "kespeech_dongbei"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/Henan.jsonl",                                             "kespeech_henan"),
    (f"{EXT_ROOT}/Zhengzhou_Dialect_Conversational_Speech_Corpus/Zhengzhou_segments_manifest.jsonl","corpus_zhengzhou"),
    (f"{EXT_ROOT}/Nanchang_Dialect_Conversational_Speech_Corpus/nanchang_segments_manifest.jsonl", "corpus_nanchang"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/Nanjing.jsonl",                                           "kespeech_nanjing"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/shan3xi.jsonl",                                           "kespeech_shan3xi"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/Shandong.jsonl",                                          "kespeech_shandong"),
    (f"{EXT_ROOT}/wenetspeech-chuan/data-50h/meta_50h_clean.jsonl",                                "wenetspeech_chuan"),
    (f"{EXT_ROOT}/Wuhan_Dialect_Scripted_Speech_Corpus/wuhan_dataset.jsonl",                       "corpus_wuhan"),
    (f"{EXT_ROOT}/wenetspeech-wu/audios_50h.jsonl",                                                "wenetspeech_wu"),
]

LABEL_TO_ACCENT: Dict[str, str] = {
    # ==== KeSpeech 系 ====
    "Anhui":     "anhui",
    "Harbin":    "dongbei",
    "Shenyang":  "dongbei",
    "Dalian":    "dongbei",
    "Changchun": "dongbei",
    "Chaoyang":  "dongbei",
    "Chifeng":   "dongbei",
    "Dandong":   "dongbei",
    "Yingkou":   "dongbei",
    "Jilin":     "dongbei",
    "Liaoning":  "dongbei",
    "Zhengzhou": "henan",
    "Nanjing":   "nanjing",
    "XiAn":      "shan3xi",
    "Xian":      "shan3xi",
    "Jinan":     "shandong",
    "Qingdao":   "shandong",
    "Weihai":    "shandong",
    "Yantai":    "shandong",
    # ==== 其他 corpus / wenetspeech ====
    "cantonese": "cantonese",
    "Cantonese": "cantonese",
    "Changsha":  "changsha",
    "Henan":     "henan",
    "Nanchang":  "nanchang",
    "Sichuan":   "sichuan",
    "wuhan":     "wuhan",
    "Wuhan":     "wuhan",
    "Wuyu":      "wuyu",
    "wuyu":      "wuyu",
}

VC_V2_TRAIN = "/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_vc_v2/train.jsonl"
VC_V2_VAL   = "/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_vc_v2/val.jsonl"
VC_V2_TEST  = "/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_vc_v2/test.jsonl"


# --------------------------- 工具函数 --------------------------- #

def build_target_text(accent: str, text: str) -> str:
    accent = (accent or "").strip()
    text = (text or "").strip()
    return f"language Chinese {accent}<asr_text>{text}"


def load_external_jsonl(path: str, source_tag: str,
                        unknown_label_counter: Counter) -> List[Dict]:
    if not os.path.isfile(path):
        print(f"[ERR] not found: {path}", file=sys.stderr)
        return []

    kept: List[Dict] = []
    n_line = 0
    n_skip_json = 0
    n_skip_label = 0
    n_skip_field = 0

    with open(path, "r", encoding="utf-8") as fin:
        for raw in fin:
            n_line += 1
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                n_skip_json += 1
                continue

            audio = obj.get("audio") or obj.get("wav_path") or ""
            text = obj.get("text") or ""
            label = obj.get("label") or obj.get("accent") or ""

            if not audio or not text:
                n_skip_field += 1
                continue

            accent = LABEL_TO_ACCENT.get(label)
            if accent is None:
                unknown_label_counter[f"{source_tag}::{label}"] += 1
                n_skip_label += 1
                continue

            key = f"ext_{source_tag}_{len(kept):07d}"
            kept.append({
                "audio":  audio,
                "text":   build_target_text(accent, text),
                "prompt": "",
                "key":    key,
                "accent": accent,
                "src":    source_tag,
            })

    print(f"[EXT] {source_tag:<20s} lines={n_line:>7d}  kept={len(kept):>7d}  "
          f"skip_json={n_skip_json}  skip_label={n_skip_label}  skip_field={n_skip_field}")
    return kept


def truncate_per_accent(samples: List[Dict], per_accent_max: int,
                        rng: random.Random) -> List[Dict]:
    """对外部数据做每 accent 硬截断（避免单一源过度主导，保留多样性）。"""
    by_accent: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_accent[s["accent"]].append(s)

    result: List[Dict] = []
    for accent, lst in sorted(by_accent.items()):
        if len(lst) > per_accent_max:
            rng.shuffle(lst)
            kept = lst[:per_accent_max]
            print(f"[TRUNC] {accent:<12s} {len(lst):>7d} -> {len(kept):>7d}")
        else:
            kept = lst
            print(f"[TRUNC] {accent:<12s} {len(lst):>7d} -> {len(kept):>7d}  (kept all)")
        result.extend(kept)
    return result


def _accent_of(obj: Dict) -> str:
    """从样本中抽取 accent 字段；若没有则从 text 里解析。"""
    if "accent" in obj and obj["accent"]:
        return obj["accent"]
    text = obj.get("text", "")
    # 期望格式: "language Chinese {accent}<asr_text>...."
    try:
        prefix = "language Chinese "
        if text.startswith(prefix):
            rest = text[len(prefix):]
            idx = rest.find("<asr_text>")
            if idx > 0:
                return rest[:idx].strip()
    except Exception:
        pass
    return "unknown"


def temperature_resample(samples: List[Dict], alpha: float,
                         total_budget: int,
                         rng: random.Random) -> List[Dict]:
    """对合并后的样本按 accent 做温度采样。

    公式：
        n_i     : accent i 的原始样本数
        weight_i = n_i ** alpha
        target_i = round( total_budget * weight_i / sum(weight_j) )
        实际抽样：从每个 accent 中不放回随机抽 min(target_i, n_i)，
                 如果 target_i > n_i 则重复上采样。

    alpha=1.0 → proportional；0.5 → 温度采样；0.0 → uniform（每 accent 完全相同）
    """
    by_accent: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_accent[_accent_of(s)].append(s)

    n_i = {a: len(lst) for a, lst in by_accent.items()}
    w_i = {a: (max(1, n) ** alpha) for a, n in n_i.items()}
    w_sum = sum(w_i.values())
    target_i = {a: max(1, int(round(total_budget * w_i[a] / w_sum))) for a in n_i}

    print(f"\n[TEMP-SAMPLE] alpha={alpha}  total_budget={total_budget}  "
          f"num_accents={len(n_i)}")
    print(f"{'accent':<12s} {'orig':>8s} {'target':>8s} {'ratio':>6s}")
    print("-" * 40)
    result: List[Dict] = []
    for a in sorted(n_i.keys()):
        lst = list(by_accent[a])
        rng.shuffle(lst)
        t = target_i[a]
        if t <= len(lst):
            picked = lst[:t]
            ratio = t / len(lst)
        else:
            # 上采样：先全部保留，剩余额度做有放回补齐
            picked = list(lst)
            deficit = t - len(lst)
            for _ in range(deficit):
                picked.append(rng.choice(lst))
            ratio = t / len(lst)
        print(f"{a:<12s} {len(lst):>8d} {t:>8d} {ratio:>6.2f}x")
        result.extend(picked)

    print(f"[TEMP-SAMPLE] total after resample = {len(result)}")
    return result


def load_vc_v2_jsonl(path: str, tag: str) -> List[Dict]:
    if not os.path.isfile(path):
        print(f"[ERR] VC_v2 file not found: {path}", file=sys.stderr)
        return []

    kept: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fin:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            obj["src"] = tag
            # 补 accent 字段（从 text 解析），方便统一采样
            if "accent" not in obj or not obj.get("accent"):
                obj["accent"] = _accent_of(obj)
            kept.append(obj)
    print(f"[VC ] {tag:<20s} loaded={len(kept):>7d}")
    return kept


def dump_jsonl(samples: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for obj in samples:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def accent_stats(samples: List[Dict]) -> Counter:
    return Counter(_accent_of(s) for s in samples)


def src_stats(samples: List[Dict]) -> Counter:
    return Counter(s.get("src", "?") for s in samples)


def format_stats(name: str, samples: List[Dict]) -> str:
    lines = [f"===== {name} (total={len(samples)}) ====="]
    lines.append("-- by accent --")
    for a, c in sorted(accent_stats(samples).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {a:<12s} {c}")
    lines.append("-- by source --")
    for s, c in sorted(src_stats(samples).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {s:<20s} {c}")
    return "\n".join(lines) + "\n"


# --------------------------- 主流程 --------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str,
                   default="/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_mixft_v4")
    p.add_argument("--per_accent_max", type=int, default=25000,
                   help="外部数据在合并前每 accent 的硬上限，避免单源污染")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="温度采样指数：1.0=按比例, 0.5=温度采样, 0.0=完全均衡")
    p.add_argument("--total_budget", type=int, default=None,
                   help="重采样后训练集总条数；不指定则用合并后总数")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. 加载全部外部方言 ---------- #
    print("\n[STEP 1] Loading external dialect jsonls ...")
    unknown_label_counter: Counter = Counter()
    ext_all: List[Dict] = []
    for path, tag in EXT_SOURCES:
        ext_all.extend(load_external_jsonl(path, tag, unknown_label_counter))

    if unknown_label_counter:
        print("\n[WARN] Unknown labels (skipped):")
        for k, v in unknown_label_counter.most_common():
            print(f"       - {k}: {v}")

    print(f"\n[STEP 1] external total loaded = {len(ext_all)}")

    # ---------- 2. 外部数据每 accent 硬截断（保多样性） ---------- #
    print(f"\n[STEP 2] Truncating external per-accent to <= {args.per_accent_max} ...")
    ext_trunc = truncate_per_accent(ext_all, args.per_accent_max, rng)
    print(f"[STEP 2] external after truncation = {len(ext_trunc)}")

    # ---------- 3. 加载 VC_v2 ---------- #
    print("\n[STEP 3] Loading VC_v2 challenge SFT data ...")
    vc_train = load_vc_v2_jsonl(VC_V2_TRAIN, "vc_v2_train")
    vc_val   = load_vc_v2_jsonl(VC_V2_VAL,   "vc_v2_val")

    # ---------- 4. 合并 & 温度采样 ---------- #
    print("\n[STEP 4] Merging external + VC_v2, then temperature resample ...")
    merged = ext_trunc + vc_train
    total_budget = args.total_budget if args.total_budget else len(merged)
    train_all = temperature_resample(merged, alpha=args.alpha,
                                     total_budget=total_budget, rng=rng)
    rng.shuffle(train_all)
    print(f"[STEP 4] final train samples = {len(train_all)}")
    print(f"[STEP 4] final val   samples = {len(vc_val)}  (VC_v2 val, unchanged)")

    # ---------- 5. 写出 ---------- #
    print("\n[STEP 5] Writing jsonl ...")
    train_out = os.path.join(out_dir, "train.jsonl")
    val_out   = os.path.join(out_dir, "val.jsonl")
    test_out  = os.path.join(out_dir, "test.jsonl")
    stats_out = os.path.join(out_dir, "stats.txt")

    dump_jsonl(train_all, train_out)
    dump_jsonl(vc_val,    val_out)

    if os.path.isfile(VC_V2_TEST):
        shutil.copyfile(VC_V2_TEST, test_out)
        print(f"[STEP 5] test.jsonl copied from {VC_V2_TEST}")
    else:
        with open(test_out, "w") as _:
            pass
        print("[STEP 5] test.jsonl written as empty placeholder")

    # ---------- 6. 统计 ---------- #
    stats_lines: List[str] = []
    stats_lines.append(format_stats("TRAIN", train_all))
    stats_lines.append(format_stats("VAL",   vc_val))
    if unknown_label_counter:
        stats_lines.append("===== Unknown labels (skipped) =====")
        for k, v in unknown_label_counter.most_common():
            stats_lines.append(f"  {k}: {v}")
        stats_lines.append("")
    stats_lines.append(f"per_accent_max = {args.per_accent_max}")
    stats_lines.append(f"alpha          = {args.alpha}")
    stats_lines.append(f"total_budget   = {total_budget}")
    stats_lines.append(f"seed           = {args.seed}")

    stats_text = "\n".join(stats_lines)
    with open(stats_out, "w", encoding="utf-8") as fout:
        fout.write(stats_text)

    print("\n" + stats_text)
    print(f"\n[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()

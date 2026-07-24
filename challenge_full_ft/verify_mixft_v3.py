#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_mixft_v3.py — 抽样验证 data_mixft_v3/train.jsonl 的格式与音频存在性。"""

import json
import os
import random

random.seed(0)

TRAIN = "/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_mixft_v3/train.jsonl"
CHALLENGE_ACCENTS = {
    "anhui", "cantonese", "changsha", "chaoshan", "dongbei", "henan",
    "kejia", "minnan", "nanchang", "nanjing", "shan1xi", "shan3xi",
    "shandong", "sichuan", "wuhan", "wuyu",
}


def main() -> None:
    lines = open(TRAIN).readlines()
    print(f"total: {len(lines)}")

    ext = [l for l in lines if '"src": "vc_v2' not in l]
    vc = [l for l in lines if '"src": "vc_v2' in l]

    print("\n===== 5 VC_v2 samples =====")
    for l in random.sample(vc, 5):
        o = json.loads(l)
        exists = os.path.isfile(o["audio"])
        print(f"accent={o['accent']:<10s} src={o['src']:<20s} exists={exists}")
        print(f"  text={o['text'][:100]}")
        print(f"  audio={o['audio'][:130]}")

    print("\n===== 15 External samples =====")
    for l in random.sample(ext, 15):
        o = json.loads(l)
        exists = os.path.isfile(o["audio"])
        print(f"accent={o['accent']:<10s} src={o['src']:<20s} exists={exists}")
        print(f"  text={o['text'][:100]}")
        print(f"  audio={o['audio'][:130]}")

    # ---- 全量校验 ---- #
    bad_prefix = [l for l in lines
                  if not json.loads(l)["text"].startswith("language Chinese ")]
    print(f"\n[CHECK] invalid text prefix (should be 0): {len(bad_prefix)}")

    bad_accent = [l for l in lines
                  if json.loads(l)["accent"] not in CHALLENGE_ACCENTS]
    print(f"[CHECK] unknown accent (should be 0): {len(bad_accent)}")
    for b in bad_accent[:3]:
        print(f"  bad: {b[:200]}")

    # ---- 音频抽样 100 条校验存在性 ---- #
    sample_100 = random.sample(lines, 100)
    n_exist = sum(1 for l in sample_100
                  if os.path.isfile(json.loads(l)["audio"]))
    print(f"\n[CHECK] audio existence in random 100: {n_exist}/100")


if __name__ == "__main__":
    main()

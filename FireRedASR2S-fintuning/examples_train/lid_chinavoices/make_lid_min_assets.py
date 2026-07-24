#!/usr/bin/env python3
"""从完整的 pretrained_model_dir（如 pretrained_models/FireRedLID）抽取 LID 推理链路
真正需要的最小离线资产：model.pth.tar 中的结构 args（权重置空，会被 finetune 出来的
checkpoint 以 strict=True 整体覆盖）+ cmvn.ark。见 pretrained_models/FireRedLID_min/README.md。
"""
import argparse
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="pretrained_models/FireRedLID",
                        help="完整 pretrained_model_dir")
    parser.add_argument("--dst", default="pretrained_models/FireRedLID_min",
                        help="输出的精简资产目录")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    pkg = torch.load(src / "model.pth.tar", map_location="cpu", weights_only=False)
    torch.save({"args": pkg["args"], "model_state_dict": {}}, dst / "model.pth.tar")
    shutil.copy(src / "cmvn.ark", dst / "cmvn.ark")
    print(f"Wrote {dst}/model.pth.tar (args only) and {dst}/cmvn.ark")


if __name__ == "__main__":
    main()

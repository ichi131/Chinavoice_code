#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "fireredasr2s"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_lid_chinavoices")

from finetune_lid_chinavoices import Collator, LABELS, LidClassifier


class SafeCollator:
    """Wraps Collator: 绝不返回 None，记录被跳过的样本。"""
    def __init__(self, cmvn_path):
        self._inner = Collator(cmvn_path)
        self.skipped_total = 0

    def __call__(self, batch):
        origin_uttids = [x["key"] for x in batch]
        origin_wavs = [x["wav_path"] for x in batch]
        result = self._inner(batch)
        if result is None:
            # 整个 batch 特征提取失败，记录并返回空标记
            self.skipped_total += len(batch)
            for key, wav in zip(origin_uttids, origin_wavs):
                logger.warning("Skipped (batch failed): key=%s wav=%s", key, wav)
            return None, None, None, origin_uttids, origin_wavs
        feats, lengths, labels, uttids, wavs = result
        # 检查 batch 内被个别跳过的样本
        if len(uttids) < len(origin_uttids):
            processed = set(uttids)
            for key, wav in zip(origin_uttids, origin_wavs):
                if key not in processed:
                    self.skipped_total += 1
                    logger.warning("Skipped (audio too short): key=%s wav=%s", key, wav)
        return result


class WavJsonlDataset(Dataset):
    def __init__(self, path):
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append({
                    "key": obj.get("key") or os.path.splitext(os.path.basename(obj["wav_path"]))[0],
                    "wav_path": obj["wav_path"],
                    "label": 0,
                    "accent": obj.get("accent", ""),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def ddp_setup():
    """Initialize DDP and return (rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        # destroy_process_group 在显存吃紧时可能抛 NCCL/CUDA OOM，属于清理阶段异常，
        # 不影响此前已写盘的推理结果；此处吞异常降级为 stderr 打印，避免让调用方（如扫描器）
        # 因为非零退出码把成功推理误判为失败。
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ddp_cleanup] torch.cuda.empty_cache failed (non-fatal): {e}", flush=True)
        try:
            dist.barrier()
        except Exception as e:
            print(f"[ddp_cleanup] dist.barrier failed (non-fatal): {e}", flush=True)
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[ddp_cleanup] dist.destroy_process_group failed (non-fatal): {e}", flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained_model_dir", default="/n/work6/yiwang/FireRedASR2S/pretrained_models/FireRedLID")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Running DDP inference on {world_size} GPU(s)")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = ckpt.get("args", {})
    model = LidClassifier(args.pretrained_model_dir, len(LABELS), freeze_encoder=True,
                          dropout=float(model_args.get("dropout", 0.1)))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    # Build dataset with DistributedSampler
    cmvn_path = os.path.join(args.pretrained_model_dir, "cmvn.ark")
    dataset = WavJsonlDataset(args.input_jsonl)
    collator = SafeCollator(cmvn_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        sampler=sampler, num_workers=args.num_workers,
                        collate_fn=collator,
                        pin_memory=torch.cuda.is_available())

    # Each rank writes to its own file
    rank_output = f"{args.output}.rank{rank}"
    os.makedirs(os.path.dirname(rank_output) or ".", exist_ok=True)
    with open(rank_output, "w", encoding="utf-8") as f:
        for batch in loader:
            feats, lengths, _, uttids, wavs = batch
            if feats is None:
                # 整个 batch 无法处理，写出 error 标记
                for u, w in zip(uttids, wavs):
                    f.write(json.dumps({
                        "key": u,
                        "wav_path": w,
                        "accent": "error",
                        "accent_cn": "错误",
                        "confidence": 0.0,
                        "top5": [],
                    }, ensure_ascii=False) + "\n")
                continue
            feats = feats.to(device)
            lengths = lengths.to(device)
            logits = model(feats, lengths)
            probs = logits.softmax(dim=-1)
            k = min(args.topk, len(LABELS))
            top_probs, top_ids = probs.topk(k=k, dim=-1)
            conf = top_probs[:, 0]
            pred = top_ids[:, 0]
            for u, w, pidx, c, sample_probs, sample_ids in zip(
                    uttids, wavs, pred.cpu().tolist(), conf.cpu().tolist(),
                    top_probs.cpu().tolist(), top_ids.cpu().tolist()):
                name, cn = LABELS[pidx]
                topk = []
                for prob, label_id in zip(sample_probs, sample_ids):
                    top_name, top_cn = LABELS[label_id]
                    topk.append({
                        "accent": top_name,
                        "accent_cn": top_cn,
                        "prob": round(float(prob), 6),
                    })
                f.write(json.dumps({
                    "key": u,
                    "wav_path": w,
                    "accent": name,
                    "accent_cn": cn,
                    "confidence": round(float(c), 6),
                    "top5": topk,
                }, ensure_ascii=False) + "\n")

    logger.info("Rank %d: skipped %d samples total (audio too short or unreadable)", rank, collator.skipped_total)

    ddp_cleanup()

    if rank == 0 and world_size > 1:
        # Merge all rank outputs
        logger.info("Merging rank outputs ...")
        with open(args.output, "w", encoding="utf-8") as fout:
            for r in range(world_size):
                part = f"{args.output}.rank{r}"
                if os.path.exists(part):
                    with open(part, encoding="utf-8") as fin:
                        fout.write(fin.read())
                    os.remove(part)
        logger.info(f"Merged {world_size} rank outputs -> {args.output}")
    elif world_size == 1:
        os.rename(rank_output, args.output)


if __name__ == "__main__":
    main()

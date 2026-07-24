#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "fireredasr2s"))

from fireredlid.data.feat import FeatExtractor
from fireredlid.lid import load_fireredlid_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("finetune_lid_chinavoices")

LABELS = [
    ("anhui", "安徽"),
    ("cantonese", "粤语"),
    ("changsha", "长沙"),
    ("chaoshan", "潮汕"),
    ("dongbei", "东北"),
    ("henan", "河南"),
    ("kejia", "客家"),
    ("minnan", "闽南"),
    ("nanchang", "南昌"),
    ("nanjing", "南京"),
    ("shan1xi", "山西"),
    ("shan3xi", "陕西"),
    ("shandong", "山东"),
    ("sichuan", "四川"),
    ("wuyu", "吴语"),
    ("wuhan", "武汉"),
]
LABEL2ID = {name: i for i, (name, _) in enumerate(LABELS)}
ID2CN = {i: cn for i, (_, cn) in enumerate(LABELS)}


class JsonlLidDataset(Dataset):
    def __init__(self, path):
        self.items = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                accent = obj["accent"]
                if accent not in LABEL2ID:
                    skipped += 1
                    continue
                self.items.append({
                    "key": obj["key"],
                    "wav_path": obj["wav_path"],
                    "label": LABEL2ID[accent],
                    "accent": accent,
                })
        if skipped:
            logger.warning("Skipped %d rows with unknown accent in %s", skipped, path)
        if not self.items:
            raise RuntimeError(f"No usable examples in {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class Collator:
    def __init__(self, cmvn_path):
        self.feat_extractor = FeatExtractor(cmvn_path)

    def __call__(self, batch):
        origin_uttids = [x["key"] for x in batch]
        origin_wavs = [x["wav_path"] for x in batch]
        origin_labels = [x["label"] for x in batch]
        feats, lengths, _, wavs, uttids = self.feat_extractor(origin_wavs, origin_uttids)
        if feats is None:
            return None
        label_by_uttid = {uttid: label for uttid, label in zip(origin_uttids, origin_labels)}
        labels = torch.tensor([label_by_uttid[u] for u in uttids], dtype=torch.long)
        return feats, lengths, labels, uttids, wavs


class LidClassifier(nn.Module):
    def __init__(self, pretrained_model_dir, num_labels, freeze_encoder=True, dropout=0.1):
        super().__init__()
        base_model = load_fireredlid_model(os.path.join(pretrained_model_dir, "model.pth.tar"))
        self.encoder = base_model.encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_model.encoder.odim, num_labels)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, feats, lengths):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                enc, _, mask = self.encoder(feats, lengths)
        else:
            enc, _, mask = self.encoder(feats, lengths)
        valid = mask.squeeze(1).to(enc.dtype)
        pooled = (enc * valid.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / valid.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        return self.classifier(self.dropout(pooled))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True, rank, local_rank, world_size, device


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def move_batch(batch, device):
    feats, lengths, labels, uttids, wavs = batch
    return feats.to(device), lengths.to(device), labels.to(device), uttids, wavs


def run_epoch(model, loader, optimizer, scheduler, scaler, device, train, use_amp, log_interval,
              distributed, rank, label_smoothing, grad_clip):
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    for step, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        feats, lengths, labels, _, _ = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(feats, lengths)
                loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
        total_loss += loss.item() * labels.numel()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        if is_main_process(rank) and train and log_interval > 0 and step % log_interval == 0:
            logger.info("train step=%d loss=%.4f acc=%.4f", step, total_loss / max(total, 1), correct / max(total, 1))
    stats = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, correct_sum, total_sum = stats.cpu().tolist()
    return loss_sum / max(total_sum, 1.0), correct_sum / max(total_sum, 1.0)


@torch.no_grad()
def dump_predictions(model, loader, device, output_path):
    model.eval()
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in loader:
            if batch is None:
                continue
            feats, lengths, labels, uttids, wavs = move_batch(batch, device)
            logits = model(feats, lengths)
            probs = logits.softmax(dim=-1)
            conf, pred = probs.max(dim=-1)
            for u, w, gold, p, c in zip(uttids, wavs, labels.cpu().tolist(), pred.cpu().tolist(), conf.cpu().tolist()):
                f.write(json.dumps({
                    "key": u,
                    "wav_path": w,
                    "gold": LABELS[gold][0],
                    "gold_cn": ID2CN[gold],
                    "pred": LABELS[p][0],
                    "pred_cn": ID2CN[p],
                    "confidence": round(float(c), 6),
                }, ensure_ascii=False) + "\n")


def save_checkpoint(path, model, optimizer, epoch, best_acc, args):
    torch.save({
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "labels": [{"id": i, "name": name, "cn": cn} for i, (name, cn) in enumerate(LABELS)],
        "args": vars(args),
    }, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", default="/n/work6/yiwang/chinavoices_challenge/scripts/data_train.jsonl")
    p.add_argument("--val_jsonl", default="/n/work6/yiwang/chinavoices_challenge/scripts/data_val.jsonl")
    p.add_argument("--pretrained_model_dir", default="/n/work6/yiwang/FireRedASR2S/pretrained_models/FireRedLID")
    p.add_argument("--init_ckpt", default="")
    p.add_argument("--output_dir", default="/n/work6/yiwang/FireRedASR2S/exp/lid_chinavoices")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--freeze_encoder", type=int, default=1)
    p.add_argument("--use_amp", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_interval", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size, device = init_distributed()
    set_seed(args.seed + rank)
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": i, "name": n, "cn": c} for i, (n, c) in enumerate(LABELS)], f, ensure_ascii=False, indent=2)
        logger.info("device=%s distributed=%s rank=%d world_size=%d labels=%s",
                    device, distributed, rank, world_size, [x[0] for x in LABELS])
    cmvn_path = os.path.join(args.pretrained_model_dir, "cmvn.ark")

    train_set = JsonlLidDataset(args.train_jsonl)
    val_set = JsonlLidDataset(args.val_jsonl)
    collator = Collator(cmvn_path)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=args.num_workers, collate_fn=collator,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler,
                            num_workers=args.num_workers, collate_fn=collator,
                            pin_memory=torch.cuda.is_available())
    if is_main_process(rank):
        logger.info("train=%d val=%d per_gpu_batch_size=%d", len(train_set), len(val_set), args.batch_size)

    model = LidClassifier(args.pretrained_model_dir, len(LABELS), bool(args.freeze_encoder), args.dropout).to(device)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if is_main_process(rank):
            logger.info("loaded init_ckpt=%s missing=%s unexpected=%s",
                        args.init_ckpt, missing, unexpected)
    if distributed:
        ddp_kwargs = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        model = DDP(model, **ddp_kwargs)
    train_model = unwrap_model(model)
    if args.encoder_lr is not None and not bool(args.freeze_encoder):
        optimizer = torch.optim.AdamW([
            {"params": [p for p in train_model.encoder.parameters() if p.requires_grad],
             "lr": args.encoder_lr},
            {"params": [p for p in train_model.classifier.parameters() if p.requires_grad],
             "lr": args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.lr, weight_decay=args.weight_decay)
    total_train_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        decay_steps = max(total_train_steps - args.warmup_steps, 1)
        progress = min(max(step - args.warmup_steps, 0) / decay_steps, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) if args.warmup_steps > 0 else None
    if is_main_process(rank):
        logger.info("optimizer lr=%s encoder_lr=%s weight_decay=%s warmup_steps=%d total_train_steps=%d",
                    args.lr, args.encoder_lr, args.weight_decay, args.warmup_steps, total_train_steps)
    use_amp = bool(args.use_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, scheduler, scaler, device, True, use_amp,
                                          args.log_interval, distributed, rank, args.label_smoothing,
                                          args.grad_clip)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, None, scaler, device, False, use_amp,
                                      0, distributed, rank, 0.0, args.grad_clip)
        if is_main_process(rank):
            logger.info("epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                        epoch, train_loss, train_acc, val_loss, val_acc)
            save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, epoch, best_acc, args)
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(args.output_dir, "best.pt")
                save_checkpoint(best_path, model, optimizer, epoch, best_acc, args)
                full_val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, collate_fn=collator,
                                             pin_memory=torch.cuda.is_available())
                dump_predictions(unwrap_model(model), full_val_loader, device, os.path.join(args.output_dir, "val_pred_best.jsonl"))
                logger.info("new best val_acc=%.4f saved=%s", best_acc, best_path)
        if distributed:
            dist.barrier()
    if is_main_process(rank):
        logger.info("done best_acc=%.4f output_dir=%s", best_acc, args.output_dir)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

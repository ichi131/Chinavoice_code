#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "fireredasr2s"))

from fireredasr2.data.asr_feat import ASRFeatExtractor
from fireredasr2.models.fireredasr_aed import FireRedAsrAed
from fireredasr2.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer

logger = logging.getLogger("finetune_asr_chinavoices")


class JsonlAsrDataset(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    key = str(obj["key"])
                    wav_path = str(obj["wav_path"])
                    text = str(obj["text"]).strip()
                    accent = str(obj["accent"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid row at {path}:{line_number}: {exc}") from exc
                if not key or not wav_path or not text or not accent:
                    raise ValueError(f"Empty required field at {path}:{line_number}")
                self.items.append({
                    "sample_id": f"{line_number}:{key}",
                    "key": key,
                    "wav_path": wav_path,
                    "text": text,
                    "accent": accent,
                })
        if not self.items:
            raise RuntimeError(f"No usable examples in {path}")
        self.accents = sorted({item["accent"] for item in self.items})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class AsrCollator:
    def __init__(self, cmvn_path, tokenizer, accent_to_id, sos_id, eos_id,
                 pad_id, max_input_frames, max_target_length):
        self.feat_extractor = ASRFeatExtractor(cmvn_path)
        self.tokenizer = tokenizer
        self.accent_to_id = accent_to_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_input_frames = max_input_frames
        self.max_target_length = max_target_length

    def __call__(self, batch):
        sample_ids = [item["sample_id"] for item in batch]
        wav_paths = [item["wav_path"] for item in batch]
        feats, lengths, _, returned_wavs, returned_ids = self.feat_extractor(
            wav_paths, sample_ids
        )
        if feats is None:
            return None

        item_by_id = {item["sample_id"]: item for item in batch}
        keep_indices = []
        token_ids = []
        items = []
        for index, sample_id in enumerate(returned_ids):
            item = item_by_id[sample_id]
            _, ids = self.tokenizer.tokenize(item["text"])
            if not ids or len(ids) > self.max_target_length:
                continue
            if self.max_input_frames > 0 and int(lengths[index]) > self.max_input_frames:
                continue
            keep_indices.append(index)
            token_ids.append(ids)
            items.append(item)
        if not keep_indices:
            return None

        index_tensor = torch.tensor(keep_indices, dtype=torch.long)
        feats = feats.index_select(0, index_tensor)
        lengths = lengths.index_select(0, index_tensor)
        max_tokens = max(len(ids) for ids in token_ids)
        targets = torch.full((len(items), max_tokens), self.pad_id, dtype=torch.long)
        decoder_inputs = torch.full(
            (len(items), max_tokens + 1), self.pad_id, dtype=torch.long
        )
        decoder_outputs = torch.full_like(decoder_inputs, self.pad_id)
        target_lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long)
        for index, ids in enumerate(token_ids):
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            targets[index, :len(ids)] = ids_tensor
            decoder_inputs[index, 0] = self.sos_id
            decoder_inputs[index, 1:len(ids) + 1] = ids_tensor
            decoder_outputs[index, :len(ids)] = ids_tensor
            decoder_outputs[index, len(ids)] = self.eos_id

        return {
            "feats": feats,
            "input_lengths": lengths,
            "targets": targets,
            "target_lengths": target_lengths,
            "decoder_inputs": decoder_inputs,
            "decoder_outputs": decoder_outputs,
            "accent_ids": torch.tensor(
                [self.accent_to_id[item["accent"]] for item in items],
                dtype=torch.long,
            ),
            "keys": [item["key"] for item in items],
            "wav_paths": [returned_wavs[index] for index in keep_indices],
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune FireRedASR2-AED on multi-dialect JSONL data."
    )
    parser.add_argument(
        "--train_jsonl",
        default="/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_train_vc.jsonl",
    )
    parser.add_argument(
        "--val_jsonl",
        default="/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_val_vc.jsonl",
    )
    parser.add_argument(
        "--pretrained_model_dir",
        default=str(PROJECT_ROOT / "pretrained_models" / "FireRedASR2-AED"),
    )
    parser.add_argument(
        "--output_dir", default=str(PROJECT_ROOT / "exp" / "asr_chinavoices_vc")
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder_lr", type=float, default=5e-6)
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--ctc_weight", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--max_input_frames", type=int, default=6000)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--save_optimizer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


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


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def move_batch(batch, device):
    tensor_keys = (
        "feats", "input_lengths", "targets", "target_lengths",
        "decoder_inputs", "decoder_outputs", "accent_ids",
    )
    for key in tensor_keys:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def edit_distance(reference, hypothesis):
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for column, hyp_token in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


def greedy_ctc_ids(log_probs, length, blank_id=0):
    frame_ids = log_probs[:length].argmax(dim=-1).tolist()
    result = []
    previous = None
    for token_id in frame_ids:
        if token_id != blank_id and token_id != previous:
            result.append(token_id)
        previous = token_id
    return result


def compute_losses(model, batch, pad_id, ctc_weight, label_smoothing):
    decoder_logits, ctc_log_probs, encoder_lengths = model(
        batch["feats"], batch["input_lengths"], batch["decoder_inputs"]
    )
    attention_loss = F.cross_entropy(
        decoder_logits.reshape(-1, decoder_logits.size(-1)),
        batch["decoder_outputs"].reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )
    ctc_loss = F.ctc_loss(
        ctc_log_probs.float().transpose(0, 1),
        batch["targets"],
        encoder_lengths,
        batch["target_lengths"],
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )
    loss = (1.0 - ctc_weight) * attention_loss + ctc_weight * ctc_loss
    return loss, attention_loss, ctc_loss, ctc_log_probs, encoder_lengths


def run_epoch(model, loader, optimizer, scheduler, scaler, device, train,
              use_amp, distributed, rank, accent_names, pad_id, args):
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    accent_stats = torch.zeros(
        len(accent_names), 2, dtype=torch.float64, device=device
    )
    num_steps = len(loader)

    for step, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        batch = move_batch(batch, device)
        should_step = train and (
            step % args.grad_accum_steps == 0 or step == num_steps
        )
        sync_context = (
            model.no_sync()
            if distributed and train and not should_step
            else nullcontext()
        )
        with sync_context:
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, attention_loss, ctc_loss, ctc_log_probs, encoder_lengths = compute_losses(
                        model, batch, pad_id, args.ctc_weight,
                        args.label_smoothing if train else 0.0,
                    )
                if train:
                    scaler.scale(loss / args.grad_accum_steps).backward()

        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_size = batch["feats"].size(0)
        totals += torch.tensor([
            loss.detach().item() * batch_size,
            attention_loss.detach().item() * batch_size,
            ctc_loss.detach().item() * batch_size,
            batch_size,
        ], dtype=torch.float64, device=device)

        if not train:
            for index in range(batch_size):
                target_length = int(batch["target_lengths"][index])
                reference = batch["targets"][index, :target_length].tolist()
                hypothesis = greedy_ctc_ids(
                    ctc_log_probs[index], int(encoder_lengths[index])
                )
                accent_id = int(batch["accent_ids"][index])
                accent_stats[accent_id, 0] += edit_distance(reference, hypothesis)
                accent_stats[accent_id, 1] += len(reference)

        if train and rank == 0 and args.log_interval > 0 and step % args.log_interval == 0:
            logger.info(
                "train step=%d/%d loss=%.4f attention=%.4f ctc=%.4f lr=%.3e",
                step, num_steps, totals[0] / totals[3].clamp_min(1),
                totals[1] / totals[3].clamp_min(1),
                totals[2] / totals[3].clamp_min(1),
                optimizer.param_groups[0]["lr"],
            )

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(accent_stats, op=dist.ReduceOp.SUM)
    count = max(totals[3].item(), 1.0)
    metrics = {
        "loss": totals[0].item() / count,
        "attention_loss": totals[1].item() / count,
        "ctc_loss": totals[2].item() / count,
    }
    if not train:
        per_accent_cer = {}
        valid_cers = []
        for accent_id, accent in enumerate(accent_names):
            errors, references = accent_stats[accent_id].tolist()
            cer = errors / max(references, 1.0)
            per_accent_cer[accent] = cer
            if references > 0:
                valid_cers.append(cer)
        metrics["cer"] = accent_stats[:, 0].sum().item() / max(
            accent_stats[:, 1].sum().item(), 1.0
        )
        metrics["macro_cer"] = sum(valid_cers) / max(len(valid_cers), 1)
        metrics["per_accent_cer"] = per_accent_cer
    return metrics


def atomic_torch_save(obj, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary_path)
    os.replace(temporary_path, path)


def save_training_checkpoint(path, model, optimizer, scheduler, scaler,
                             model_args, epoch, best_macro_cer, args, metrics):
    package = {
        "args": model_args,
        "model_state_dict": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "best_macro_cer": best_macro_cer,
        "metrics": metrics,
        "finetune_args": vars(args),
    }
    if args.save_optimizer:
        package.update({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        })
    atomic_torch_save(package, path)


def save_inference_model(path, model, model_args, metrics, args):
    atomic_torch_save({
        "args": model_args,
        "model_state_dict": unwrap_model(model).state_dict(),
        "metrics": metrics,
        "finetune_args": vars(args),
    }, path)


def copy_inference_assets(model_dir, output_dir):
    for name in ("cmvn.ark", "dict.txt", "train_bpe1000.model"):
        source = Path(model_dir) / name
        destination = Path(output_dir) / name
        if not destination.exists():
            shutil.copy2(source, destination)


def validate_paths(args):
    required = [
        Path(args.train_jsonl),
        Path(args.val_jsonl),
        Path(args.pretrained_model_dir) / "model.pth.tar",
        Path(args.pretrained_model_dir) / "cmvn.ark",
        Path(args.pretrained_model_dir) / "dict.txt",
        Path(args.pretrained_model_dir) / "train_bpe1000.model",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if not 0.0 <= args.ctc_weight <= 1.0:
        raise ValueError("--ctc_weight must be in [0, 1]")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be at least 1")


def main():
    args = parse_args()
    validate_paths(args)
    distributed, rank, local_rank, world_size, device = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(args.seed + rank)

    model_path = Path(args.pretrained_model_dir) / "model.pth.tar"
    base_package = torch.load(model_path, map_location="cpu", weights_only=False)
    model_args = base_package["args"]
    model = FireRedAsrAed.from_args(model_args)
    model.load_state_dict(base_package["model_state_dict"], strict=True)
    del base_package
    tokenizer = ChineseCharEnglishSpmTokenizer(
        str(Path(args.pretrained_model_dir) / "dict.txt"),
        str(Path(args.pretrained_model_dir) / "train_bpe1000.model"),
    )
    if len(tokenizer.dict) != model_args.odim:
        raise ValueError(
            f"Vocabulary mismatch: dict={len(tokenizer.dict)} model={model_args.odim}"
        )

    train_set = JsonlAsrDataset(args.train_jsonl)
    val_set = JsonlAsrDataset(args.val_jsonl)
    accent_names = sorted(set(train_set.accents) | set(val_set.accents))
    accent_to_id = {accent: index for index, accent in enumerate(accent_names)}
    collator = AsrCollator(
        str(Path(args.pretrained_model_dir) / "cmvn.ark"),
        tokenizer, accent_to_id, model.sos_id, model.eos_id,
        model.decoder.pad_id, args.max_input_frames, args.max_target_length,
    )

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed,
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False,
    ) if distributed else None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, shuffle=train_sampler is None, sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set, shuffle=False, sampler=val_sampler, **loader_kwargs,
    )

    resume_package = None
    if args.resume:
        resume_package = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_package["model_state_dict"], strict=True)
    model.to(device)
    if distributed:
        ddp_kwargs = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        model = DDP(model, **ddp_kwargs)

    train_model = unwrap_model(model)
    encoder_parameters = [p for p in train_model.encoder.parameters() if p.requires_grad]
    decoder_parameters = [
        p for name, p in train_model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.encoder_lr},
        {"params": decoder_parameters, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = max(updates_per_epoch * args.epochs, 1)

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        decay_steps = max(total_updates - args.warmup_steps, 1)
        progress = min(max(step - args.warmup_steps, 0) / decay_steps, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = bool(args.use_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 1
    best_macro_cer = float("inf")
    if resume_package is not None:
        start_epoch = int(resume_package.get("epoch", 0)) + 1
        best_macro_cer = float(resume_package.get("best_macro_cer", float("inf")))
        if "optimizer_state_dict" in resume_package:
            optimizer.load_state_dict(resume_package["optimizer_state_dict"])
            scheduler.load_state_dict(resume_package["scheduler_state_dict"])
            scaler.load_state_dict(resume_package["scaler_state_dict"])

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        copy_inference_assets(args.pretrained_model_dir, output_dir)
        with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        with open(output_dir / "accents.json", "w", encoding="utf-8") as f:
            json.dump(accent_names, f, ensure_ascii=False, indent=2)
        parameter_count = sum(p.numel() for p in train_model.parameters())
        trainable_count = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
        logger.info(
            "device=%s world_size=%d train=%d val=%d accents=%s",
            device, world_size, len(train_set), len(val_set), accent_names,
        )
        logger.info(
            "parameters=%d trainable=%d per_gpu_batch=%d grad_accum=%d effective_batch=%d",
            parameter_count, trainable_count, args.batch_size,
            args.grad_accum_steps,
            args.batch_size * args.grad_accum_steps * world_size,
        )
    if distributed:
        dist.barrier()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            True, use_amp, distributed, rank, accent_names,
            train_model.decoder.pad_id, args,
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, scheduler, scaler, device,
            False, use_amp, distributed, rank, accent_names,
            train_model.decoder.pad_id, args,
        )
        if rank == 0:
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f val_ctc_cer=%.2f%% macro_ctc_cer=%.2f%%",
                epoch, train_metrics["loss"], val_metrics["loss"],
                100.0 * val_metrics["cer"], 100.0 * val_metrics["macro_cer"],
            )
            logger.info(
                "per_accent_ctc_cer=%s",
                {name: round(100.0 * cer, 2) for name, cer in val_metrics["per_accent_cer"].items()},
            )
            epoch_metrics = {"train": train_metrics, "validation": val_metrics}
            is_best = val_metrics["macro_cer"] < best_macro_cer
            if is_best:
                best_macro_cer = val_metrics["macro_cer"]
            save_training_checkpoint(
                output_dir / "last.pt", model, optimizer, scheduler, scaler,
                model_args, epoch, best_macro_cer, args, epoch_metrics,
            )
            if is_best:
                save_inference_model(
                    output_dir / "model.pth.tar", model, model_args,
                    epoch_metrics, args,
                )
                with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {"epoch": epoch, **epoch_metrics}, f,
                        ensure_ascii=False, indent=2,
                    )
                logger.info("saved new best inference model to %s", output_dir)
        if distributed:
            dist.barrier()

    if rank == 0:
        logger.info("done best_macro_ctc_cer=%.2f%%", 100.0 * best_macro_cer)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

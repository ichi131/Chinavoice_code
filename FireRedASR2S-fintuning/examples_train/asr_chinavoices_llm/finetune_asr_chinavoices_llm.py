#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import logging
import math
import os
import random
import shutil
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "fireredasr2s"))

try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    pkg_resources = types.ModuleType("pkg_resources")
    pkg_resources.get_distribution = importlib.metadata.distribution
    sys.modules["pkg_resources"] = pkg_resources

from fireredasr2.data.asr_feat import ASRFeatExtractor
from fireredasr2.models.fireredasr_llm import FireRedAsrLlm
from fireredasr2.tokenizer.llm_tokenizer import (
    DEFAULT_SPEECH_TOKEN,
    IGNORE_TOKEN_ID,
    LlmTokenizerWrapper,
)

logger = logging.getLogger("finetune_asr_chinavoices_llm")


class JsonlAsrDataset(Dataset):
    def __init__(self, path):
        self.items = []
        with open(path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    key = str(row["key"]).strip()
                    wav_path = str(row["wav_path"]).strip()
                    text = str(row["text"]).strip()
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid row at {path}:{line_number}: {exc}") from exc
                if not key or not wav_path or not text:
                    raise ValueError(f"Empty required field at {path}:{line_number}")
                self.items.append({
                    "sample_id": f"{line_number}:{key}",
                    "key": key,
                    "wav_path": wav_path,
                    "text": text,
                })
        if not self.items:
            raise RuntimeError(f"No usable examples in {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class LlmAsrCollator:
    def __init__(self, cmvn_path, tokenizer, max_input_frames, max_text_length):
        self.feat_extractor = ASRFeatExtractor(cmvn_path)
        self.tokenizer = tokenizer
        self.max_input_frames = max_input_frames
        self.max_text_length = max_text_length
        self.speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)

    def __call__(self, batch):
        sample_ids = [item["sample_id"] for item in batch]
        wav_paths = [item["wav_path"] for item in batch]
        feats, lengths, _, returned_wavs, returned_ids = self.feat_extractor(
            wav_paths, sample_ids
        )
        if feats is None:
            raise RuntimeError("All audio files in a batch failed feature extraction")

        item_by_id = {item["sample_id"]: item for item in batch}
        keep_indices = []
        items = []
        for index, sample_id in enumerate(returned_ids):
            if self.max_input_frames > 0 and int(lengths[index]) > self.max_input_frames:
                continue
            keep_indices.append(index)
            items.append(item_by_id[sample_id])
        if not keep_indices:
            raise RuntimeError(
                "All samples in a batch exceed --max_input_frames; "
                "filter the JSONL or increase the limit"
            )

        input_ids, attention_mask, labels, clean_texts = \
            LlmTokenizerWrapper.preprocess_texts(
                [item["text"] for item in items],
                self.tokenizer,
                self.max_text_length,
                decode=False,
            )
        speech_counts = input_ids.eq(self.speech_token_id).sum(dim=1)
        if not torch.all(speech_counts.eq(1)):
            raise RuntimeError("Each training prompt must contain exactly one <speech> token")
        if not torch.all(labels.ne(IGNORE_TOKEN_ID).any(dim=1)):
            raise RuntimeError(
                "A transcript was fully truncated; increase --max_text_length"
            )

        index_tensor = torch.tensor(keep_indices, dtype=torch.long)
        return {
            "feats": feats.index_select(0, index_tensor),
            "input_lengths": lengths.index_select(0, index_tensor),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "keys": [item["key"] for item in items],
            "wav_paths": [returned_wavs[index] for index in keep_indices],
            "clean_texts": clean_texts,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune FireRedASR2-LLM on ASR JSONL data with Adapter+LoRA."
    )
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument(
        "--pretrained_model_dir",
        default=str(PROJECT_ROOT / "pretrained_models" / "FireRedASR2-LLM"),
    )
    parser.add_argument(
        "--output_dir", default=str(PROJECT_ROOT / "exp" / "asr_chinavoices_llm")
    )
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--train_mode", choices=("adapter_lora", "adapter_only"),
        default="adapter_lora",
    )
    parser.add_argument("--freeze_encoder", type=int, choices=(0, 1), default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_input_frames", type=int, default=4000)
    parser.add_argument("--max_text_length", type=int, default=256)
    parser.add_argument("--use_amp", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use_flash_attn", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--gradient_checkpointing", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--save_optimizer", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_interval", type=int, default=20)
    return parser.parse_args()


def validate_args(args):
    model_dir = Path(args.pretrained_model_dir)
    required_files = [
        Path(args.train_jsonl),
        Path(args.val_jsonl),
        model_dir / "model.pth.tar",
        model_dir / "asr_encoder.pth.tar",
        model_dir / "cmvn.ark",
        model_dir / "Qwen2-7B-Instruct" / "config.json",
        model_dir / "Qwen2-7B-Instruct" / "model.safetensors.index.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be at least 1")
    if args.max_text_length < 32:
        raise ValueError("--max_text_length must be at least 32")


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


def add_lora(llm):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "gate_proj", "down_proj",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(llm, config)


def build_model(args):
    model_dir = Path(args.pretrained_model_dir)
    package = torch.load(
        model_dir / "model.pth.tar", map_location="cpu", weights_only=False
    )
    model_args = package["args"]
    pretrained_state = package["model_state_dict"]
    checkpoint_has_lora = any(
        name.startswith("llm.") and "lora_" in name
        for name in pretrained_state
    )
    model_args.encoder_path = str(model_dir / "asr_encoder.pth.tar")
    model_args.llm_dir = str(model_dir / "Qwen2-7B-Instruct")
    model_args.freeze_encoder = bool(args.freeze_encoder)
    model_args.freeze_llm = False
    model_args.use_lora = checkpoint_has_lora
    model_args.use_flash_attn = bool(args.use_flash_attn)
    model_args.use_fp16 = bool(args.use_amp)

    logger.info(
        "pretrained checkpoint LoRA structure: %s", checkpoint_has_lora
    )
    model = FireRedAsrLlm.from_args(model_args)
    for prefix in ("encoder.", "encoder_projector."):
        if not any(name.startswith(prefix) for name in pretrained_state):
            raise RuntimeError(
                f"Pretrained checkpoint contains no {prefix} weights"
            )
    incompatible = model.load_state_dict(pretrained_state, strict=False)
    critical_missing = [
        name for name in incompatible.missing_keys
        if name.startswith(("encoder.", "encoder_projector."))
    ]
    if critical_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Pretrained checkpoint is incompatible: "
            f"critical_missing={critical_missing[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    logger.info(
        "loaded pretrained FireRedASR2-LLM; omitted base LLM keys=%d",
        len(incompatible.missing_keys),
    )
    del package

    for parameter in model.llm.parameters():
        parameter.requires_grad = False
    if args.train_mode == "adapter_lora":
        if checkpoint_has_lora:
            lora_parameters = [
                parameter for name, parameter in model.llm.named_parameters()
                if "lora_" in name
            ]
            if not lora_parameters:
                raise RuntimeError(
                    "Checkpoint has LoRA weights, but the model has no LoRA parameters"
                )
            for parameter in lora_parameters:
                parameter.requires_grad = True
            logger.info(
                "continuing fine-tuning of %d pretrained LoRA tensors",
                len(lora_parameters),
            )
        else:
            model.llm = add_lora(model.llm)
            logger.info("initialized new LoRA layers")
        model.freeze_llm = False
        model_args.freeze_llm = False
        model_args.use_lora = True
    else:
        model.llm.eval()
        model.freeze_llm = True
        model_args.freeze_llm = not checkpoint_has_lora
        model_args.use_lora = checkpoint_has_lora
    return model, model_args


def move_batch(batch, device):
    for key in ("feats", "input_lengths", "input_ids", "attention_mask", "labels"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def run_epoch(model, loader, optimizer, scheduler, device, train, args,
              distributed, rank, trainable_parameters):
    model.train(train)
    raw_model = unwrap_model(model)
    if raw_model.freeze_encoder:
        raw_model.encoder.eval()
    if raw_model.freeze_llm:
        raw_model.llm.eval()
    if train:
        optimizer.zero_grad(set_to_none=True)

    totals = torch.zeros(2, dtype=torch.float64, device=device)
    num_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
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
                with autocast_context(device, bool(args.use_amp)):
                    outputs = model(
                        batch["feats"],
                        batch["input_lengths"],
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["labels"],
                    )
                    loss = outputs.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
                if train:
                    (loss / args.grad_accum_steps).backward()

        if should_step:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_size = batch["feats"].size(0)
        totals[0] += loss.detach().double() * batch_size
        totals[1] += batch_size
        if train and rank == 0 and args.log_interval > 0 \
                and step % args.log_interval == 0:
            logger.info(
                "train step=%d/%d loss=%.4f lr=%s",
                step,
                num_steps,
                (totals[0] / totals[1].clamp_min(1)).item(),
                [f"{group['lr']:.3e}" for group in optimizer.param_groups],
            )

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {"loss": (totals[0] / totals[1].clamp_min(1)).item()}


def make_optimizer(model, args):
    groups = []
    raw_model = unwrap_model(model)
    encoder_parameters = [
        parameter for parameter in raw_model.encoder.parameters()
        if parameter.requires_grad
    ]
    adapter_parameters = [
        parameter for parameter in raw_model.encoder_projector.parameters()
        if parameter.requires_grad
    ]
    llm_parameters = [
        parameter for parameter in raw_model.llm.parameters()
        if parameter.requires_grad
    ]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": args.encoder_lr})
    if adapter_parameters:
        groups.append({"params": adapter_parameters, "lr": args.adapter_lr})
    if llm_parameters:
        groups.append({"params": llm_parameters, "lr": args.lora_lr})
    if not groups:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    trainable_parameters = [
        parameter for parameter in raw_model.parameters() if parameter.requires_grad
    ]
    return optimizer, trainable_parameters


def cpu_state_dict(model, predicate):
    raw_model = unwrap_model(model)
    return {
        name: tensor.detach().cpu()
        for name, tensor in raw_model.state_dict().items()
        if predicate(name)
    }


def trainable_state_dict(model):
    raw_model = unwrap_model(model)
    trainable_names = {
        name for name, parameter in raw_model.named_parameters()
        if parameter.requires_grad
    }
    return cpu_state_dict(model, lambda name: name in trainable_names)


def inference_state_dict(model):
    return cpu_state_dict(
        model,
        lambda name: not name.startswith("llm.") or "lora_" in name,
    )


def atomic_torch_save(package, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(package, temporary_path)
    os.replace(temporary_path, path)


def save_training_checkpoint(path, model, optimizer, scheduler, model_args,
                             epoch, best_val_loss, metrics, args):
    package = {
        "args": model_args,
        "model_state_dict": trainable_state_dict(model),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "finetune_args": vars(args),
    }
    if args.save_optimizer:
        package["optimizer_state_dict"] = optimizer.state_dict()
        package["scheduler_state_dict"] = scheduler.state_dict()
    atomic_torch_save(package, path)


def save_inference_model(path, model, model_args, metrics, args):
    atomic_torch_save({
        "args": model_args,
        "model_state_dict": inference_state_dict(model),
        "metrics": metrics,
        "finetune_args": vars(args),
    }, path)


def prepare_inference_assets(model_dir, output_dir):
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir)
    for name in ("cmvn.ark", "asr_encoder.pth.tar"):
        destination = output_dir / name
        if not destination.exists():
            shutil.copy2(model_dir / name, destination)

    qwen_source = model_dir / "Qwen2-7B-Instruct"
    qwen_destination = output_dir / "Qwen2-7B-Instruct"
    if qwen_destination.exists() or qwen_destination.is_symlink():
        if qwen_destination.resolve() != qwen_source:
            raise FileExistsError(
                f"Existing Qwen path points elsewhere: {qwen_destination}"
            )
    else:
        qwen_destination.symlink_to(
            os.path.relpath(qwen_source, output_dir), target_is_directory=True
        )


def validate_resume(package, args):
    saved_args = package.get("finetune_args", {})
    for name in ("train_mode", "freeze_encoder"):
        if name in saved_args and saved_args[name] != getattr(args, name):
            raise ValueError(
                f"Resume mismatch for {name}: checkpoint={saved_args[name]!r}, "
                f"current={getattr(args, name)!r}"
            )


def main():
    args = parse_args()
    validate_args(args)
    distributed, rank, local_rank, world_size, device = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(args.seed + rank)

    model, model_args = build_model(args)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(
        model_args.llm_dir, use_flash_attn=bool(args.use_flash_attn)
    )
    speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    if speech_token_id >= model.llm.get_input_embeddings().num_embeddings:
        raise ValueError(
            f"Speech token id {speech_token_id} exceeds embedding vocabulary"
        )

    resume_package = None
    if args.resume:
        resume_package = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(resume_package, args)
        resume_state = resume_package["model_state_dict"]
        expected_keys = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        actual_keys = set(resume_state)
        if actual_keys != expected_keys:
            raise RuntimeError(
                "Resume checkpoint trainable keys do not match the current model: "
                f"missing={sorted(expected_keys - actual_keys)[:20]}, "
                f"unexpected={sorted(actual_keys - expected_keys)[:20]}"
            )
        incompatible = model.load_state_dict(resume_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected resume keys: {incompatible.unexpected_keys[:20]}"
            )
        logger.info("loaded resume delta with %d tensors", len(resume_state))

    train_set = JsonlAsrDataset(args.train_jsonl)
    val_set = JsonlAsrDataset(args.val_jsonl)
    collator = LlmAsrCollator(
        str(Path(args.pretrained_model_dir) / "cmvn.ark"),
        tokenizer,
        args.max_input_frames,
        args.max_text_length,
    )
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False
    ) if distributed else None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, shuffle=train_sampler is None, sampler=train_sampler, **loader_kwargs
    )
    val_loader = DataLoader(
        val_set, shuffle=False, sampler=val_sampler, **loader_kwargs
    )

    if args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.llm.config.use_cache = False
    model.to(device)
    if distributed:
        ddp_kwargs = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        model = DDP(model, **ddp_kwargs)

    optimizer, trainable_parameters = make_optimizer(model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = max(updates_per_epoch * args.epochs, 1)

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        decay_steps = max(total_updates - args.warmup_steps, 1)
        progress = min(max(step - args.warmup_steps, 0) / decay_steps, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch = 1
    best_val_loss = float("inf")
    if resume_package is not None:
        if "optimizer_state_dict" not in resume_package \
                or "scheduler_state_dict" not in resume_package:
            raise RuntimeError(
                "--resume requires a checkpoint saved with --save_optimizer 1"
            )
        start_epoch = int(resume_package.get("epoch", 0)) + 1
        best_val_loss = float(resume_package.get("best_val_loss", float("inf")))
        optimizer.load_state_dict(resume_package["optimizer_state_dict"])
        scheduler.load_state_dict(resume_package["scheduler_state_dict"])
        del resume_package

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        prepare_inference_assets(args.pretrained_model_dir, output_dir)
        with open(output_dir / "train_config.json", "w", encoding="utf-8") as target:
            json.dump(vars(args), target, ensure_ascii=False, indent=2)
        raw_model = unwrap_model(model)
        parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
        trainable_count = sum(
            parameter.numel() for parameter in raw_model.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "device=%s world_size=%d train=%d val=%d mode=%s",
            device, world_size, len(train_set), len(val_set), args.train_mode,
        )
        logger.info(
            "parameters=%d trainable=%d per_gpu_batch=%d grad_accum=%d effective_batch=%d",
            parameter_count,
            trainable_count,
            args.batch_size,
            args.grad_accum_steps,
            args.batch_size * args.grad_accum_steps * world_size,
        )
        logger.info("output_dir=%s", output_dir)
    if distributed:
        dist.barrier()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, optimizer, scheduler, device, True, args,
            distributed, rank, trainable_parameters,
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, scheduler, device, False, args,
            distributed, rank, trainable_parameters,
        )
        if rank == 0:
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f",
                epoch, train_metrics["loss"], val_metrics["loss"],
            )
            metrics = {"train": train_metrics, "validation": val_metrics}
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
            save_training_checkpoint(
                output_dir / "last.pt", model, optimizer, scheduler, model_args,
                epoch, best_val_loss, metrics, args,
            )
            if is_best:
                save_inference_model(
                    output_dir / "model.pth.tar", model, model_args, metrics, args
                )
                with open(
                    output_dir / "best_metrics.json", "w", encoding="utf-8"
                ) as target:
                    json.dump(
                        {"epoch": epoch, **metrics}, target,
                        ensure_ascii=False, indent=2,
                    )
                logger.info("saved new best inference model to %s", output_dir)
        if distributed:
            dist.barrier()

    if rank == 0:
        logger.info("done best_val_loss=%.4f", best_val_loss)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

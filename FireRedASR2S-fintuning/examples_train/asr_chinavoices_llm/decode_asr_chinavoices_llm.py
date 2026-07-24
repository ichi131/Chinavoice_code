#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "fireredasr2s"))

try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    pkg_resources = types.ModuleType("pkg_resources")
    pkg_resources.get_distribution = importlib.metadata.distribution
    sys.modules["pkg_resources"] = pkg_resources

from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

logger = logging.getLogger("decode_asr_chinavoices_llm")

ASR_MARKER = "<asr_text>"
DIALECT_RE = re.compile(r"language\s+Chinese\s+([^\s<]+)", re.IGNORECASE)
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "exp" / "asr_chinavoices_llm_lora_20260720_140508"
)
DEFAULT_INPUT_JSONL = Path(
    "/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/"
    "user_ichiwang/workspace/challenge_full_ft/data/test.jsonl"
)
DEFAULT_OUTPUT_JSONL = DEFAULT_MODEL_DIR / "pred_test.jsonl"


def split_reference(content):
    content = str(content or "").strip()
    if ASR_MARKER in content:
        prefix, text = content.split(ASR_MARKER, 1)
    else:
        prefix, text = "", content
    match = DIALECT_RE.search(prefix)
    return {
        "full": content,
        "text": text.strip(),
        "dialect": match.group(1).strip() if match else "",
    }


def load_samples(path, limit=0):
    samples = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
                audio_path = str(row["audio"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid JSONL row at {path}:{line_number}: {exc}"
                ) from exc
            if not audio_path:
                raise ValueError(f"Empty audio path at {path}:{line_number}")

            key = str(
                row.get("key") or row.get("id") or Path(audio_path).stem
            ).strip()
            if not key:
                raise ValueError(f"Empty utterance key at {path}:{line_number}")

            reference = split_reference(row.get("text", ""))
            dialect = str(row.get("accent") or reference["dialect"]).strip()
            samples.append({
                "decode_id": f"{line_number}:{key}",
                "utt_id": key,
                "audio_path": audio_path,
                "ref_full": reference["full"],
                "ref_text": reference["text"],
                "ref_dialect": dialect,
            })
            if limit > 0 and len(samples) >= limit:
                break

    if not samples:
        raise RuntimeError(f"No usable samples found in {path}")
    return samples


def build_pred_full(dialect, text):
    if dialect:
        return f"language Chinese {dialect}{ASR_MARKER}{text}"
    return f"{ASR_MARKER}{text}"


def empty_result(sample, error):
    dialect = sample["ref_dialect"]
    return {
        "utt_id": sample["utt_id"],
        "audio_path": sample["audio_path"],
        "ref_full": sample["ref_full"],
        "ref_text": sample["ref_text"],
        "ref_dialect": dialect,
        "pred_full": build_pred_full(dialect, ""),
        "pred_text": "",
        "pred_dialect": dialect,
        "error": error,
    }


def format_result(sample, decoded):
    text = str(decoded.get("text") or "").strip()
    if not text:
        return empty_result(sample, "empty transcription or audio decode failed")

    dialect = sample["ref_dialect"]
    result = {
        "utt_id": sample["utt_id"],
        "audio_path": sample["audio_path"],
        "ref_full": sample["ref_full"],
        "ref_text": sample["ref_text"],
        "ref_dialect": dialect,
        "pred_full": build_pred_full(dialect, text),
        "pred_text": text,
        "pred_dialect": dialect,
        "error": "",
    }
    for name in ("dur_s", "rtf"):
        if name in decoded:
            result[name] = decoded[name]
    return result


def transcribe(model, samples):
    decode_ids = [sample["decode_id"] for sample in samples]
    audio_paths = [sample["audio_path"] for sample in samples]
    decoded = model.transcribe(decode_ids, audio_paths)
    return {item["uttid"]: item for item in decoded}


def decode_batch(model, samples):
    valid_samples = []
    output_by_id = {}
    for sample in samples:
        if not os.path.isfile(sample["audio_path"]):
            output_by_id[sample["decode_id"]] = empty_result(
                sample, "audio file not found"
            )
        else:
            valid_samples.append(sample)

    if valid_samples:
        decoded_by_id = transcribe(model, valid_samples)
        retry_samples = []
        for sample in valid_samples:
            decoded = decoded_by_id.get(sample["decode_id"])
            if decoded is None or not str(decoded.get("text") or "").strip():
                retry_samples.append(sample)
            else:
                output_by_id[sample["decode_id"]] = format_result(sample, decoded)

        for sample in retry_samples:
            decoded = transcribe(model, [sample]).get(sample["decode_id"])
            if decoded is None:
                output_by_id[sample["decode_id"]] = empty_result(
                    sample, "model returned no result"
                )
            else:
                output_by_id[sample["decode_id"]] = format_result(sample, decoded)

    return [output_by_id[sample["decode_id"]] for sample in samples]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decode ChinaVoices JSONL with a fine-tuned "
            "FireRedASR2-LLM LoRA model."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu-ids",
        type=str,
        help="Comma-separated GPU indices, such as 0,1,2,3; use 'all' for all visible GPUs",
    )
    device_group.add_argument(
        "--gpu",
        type=int,
        help="Single GPU index for backward compatibility",
    )
    parser.add_argument("--use-half", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size per GPU (default: 1)",
    )
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument(
        "--decode-max-len",
        type=int,
        default=0,
        help="Maximum generated tokens; 0 derives the limit from audio length",
    )
    parser.add_argument("--decode-min-len", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=3.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Decode only N rows; 0 means all")
    return parser.parse_args()


def resolve_gpu_ids(args):
    if args.gpu is not None:
        gpu_ids = [args.gpu]
    elif args.gpu_ids is None:
        gpu_ids = [0]
    elif args.gpu_ids.strip().lower() == "all":
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        try:
            gpu_ids = [
                int(item.strip())
                for item in args.gpu_ids.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "--gpu-ids must be comma-separated integers or 'all'"
            ) from exc

    if not gpu_ids:
        raise ValueError("--gpu-ids selected no GPUs")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU indices must be non-negative")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU indices must not contain duplicates")
    return gpu_ids


def validate_args(args, gpu_ids):
    required_assets = (
        args.model_dir / "model.pth.tar",
        args.model_dir / "asr_encoder.pth.tar",
        args.model_dir / "cmvn.ark",
        args.model_dir / "Qwen2-7B-Instruct" / "config.json",
        args.model_dir / "Qwen2-7B-Instruct" / "model.safetensors.index.json",
    )
    missing = [str(path) for path in required_assets if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model assets:\n" + "\n".join(missing))
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.beam_size < 1:
        raise ValueError("--beam-size must be >= 1")
    if args.decode_max_len < 0 or args.decode_min_len < 0:
        raise ValueError("decode length limits must be >= 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this decoder on a GPU node")
    device_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= device_count]
    if invalid:
        raise ValueError(
            f"GPU indices {invalid} are unavailable; visible GPU count is {device_count}"
        )


def create_model(args, gpu_id):
    torch.cuda.set_device(gpu_id)
    config = FireRedAsr2Config(
        use_gpu=True,
        use_half=args.use_half,
        beam_size=args.beam_size,
        decode_max_len=args.decode_max_len,
        decode_min_len=args.decode_min_len,
        repetition_penalty=args.repetition_penalty,
        llm_length_penalty=args.length_penalty,
        temperature=args.temperature,
    )
    return FireRedAsr2.from_pretrained("llm", str(args.model_dir), config)


def shard_path(shard_dir, rank):
    return Path(shard_dir) / f"rank_{rank:05d}.jsonl"


def decode_worker(rank, gpu_ids, args, samples, shard_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    gpu_id = gpu_ids[rank]
    local_samples = samples[rank::len(gpu_ids)]
    logger.info(
        "rank=%d gpu=%d samples=%d batch_size=%d loading model",
        rank,
        gpu_id,
        len(local_samples),
        args.batch_size,
    )
    model = create_model(args, gpu_id)

    completed = 0
    error_count = 0
    last_logged = 0
    final_path = shard_path(shard_dir, rank)
    temporary_path = final_path.with_suffix(final_path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for start in range(0, len(local_samples), args.batch_size):
                batch = local_samples[start:start + args.batch_size]
                results = decode_batch(model, batch)
                for sample, result in zip(batch, results):
                    output.write(json.dumps({
                        "input_index": sample["input_index"],
                        "result": result,
                    }, ensure_ascii=False) + "\n")
                    error_count += int(bool(result["error"]))
                output.flush()
                completed += len(batch)
                if (
                    completed == len(local_samples)
                    or (
                        args.log_interval > 0
                        and completed - last_logged >= args.log_interval
                    )
                ):
                    logger.info(
                        "rank=%d gpu=%d progress=%d/%d errors=%d",
                        rank,
                        gpu_id,
                        completed,
                        len(local_samples),
                        error_count,
                    )
                    last_logged = completed
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    logger.info(
        "rank=%d gpu=%d finished samples=%d errors=%d",
        rank,
        gpu_id,
        len(local_samples),
        error_count,
    )


def merge_shards(shard_dir, num_shards, output_jsonl, expected_count):
    results_by_index = {}
    for rank in range(num_shards):
        part_path = shard_path(shard_dir, rank)
        if not part_path.is_file():
            raise RuntimeError(f"Missing decoder shard: {part_path}")
        with part_path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                try:
                    record = json.loads(raw_line)
                    input_index = int(record["input_index"])
                    result = record["result"]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid decoder shard row at {part_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(result, dict):
                    raise RuntimeError(
                        f"Invalid result object at {part_path}:{line_number}"
                    )
                if input_index in results_by_index:
                    raise RuntimeError(f"Duplicate decoded input index: {input_index}")
                results_by_index[input_index] = result

    missing = [index for index in range(expected_count) if index not in results_by_index]
    unexpected = [index for index in results_by_index if not 0 <= index < expected_count]
    if missing or unexpected:
        raise RuntimeError(
            f"Decoded result mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    merged_path = Path(shard_dir) / "merged.jsonl"
    error_count = 0
    with merged_path.open("w", encoding="utf-8") as output:
        for input_index in range(expected_count):
            result = results_by_index[input_index]
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            error_count += int(bool(result["error"]))
    os.replace(merged_path, output_jsonl)
    return error_count


def main():
    args = parse_args()
    gpu_ids = resolve_gpu_ids(args)
    validate_args(args, gpu_ids)

    samples = load_samples(args.input_jsonl, args.limit)
    for input_index, sample in enumerate(samples):
        sample["input_index"] = input_index
    if len(gpu_ids) > len(samples):
        logger.warning(
            "Reducing decoder workers from %d to %d because there are fewer samples",
            len(gpu_ids),
            len(samples),
        )
        gpu_ids = gpu_ids[:len(samples)]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    logger.info("model_dir=%s", args.model_dir)
    logger.info("input_jsonl=%s samples=%d", args.input_jsonl, len(samples))
    logger.info("output_jsonl=%s", args.output_jsonl)
    logger.info(
        "gpu_ids=%s workers=%d batch_size_per_gpu=%d effective_batch_size=%d",
        gpu_ids,
        len(gpu_ids),
        args.batch_size,
        len(gpu_ids) * args.batch_size,
    )

    shard_dir = Path(tempfile.mkdtemp(
        prefix=f".{args.output_jsonl.name}.parts.",
        dir=str(args.output_jsonl.parent),
    ))
    try:
        if len(gpu_ids) == 1:
            decode_worker(0, gpu_ids, args, samples, shard_dir)
        else:
            mp.spawn(
                decode_worker,
                args=(gpu_ids, args, samples, shard_dir),
                nprocs=len(gpu_ids),
                join=True,
            )
        error_count = merge_shards(
            shard_dir,
            len(gpu_ids),
            args.output_jsonl,
            len(samples),
        )
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)

    logger.info(
        "decode finished samples=%d errors=%d output=%s",
        len(samples),
        error_count,
        args.output_jsonl,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()

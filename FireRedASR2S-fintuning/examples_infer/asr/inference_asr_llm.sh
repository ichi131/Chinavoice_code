#!/bin/bash

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Cannot find conda. Please make sure conda is available before running this script." >&2
    exit 1
fi
conda activate fireredasr2s

export PATH=$PWD/fireredasr2/:$PWD/fireredasr2/utils/:${PATH:-}
export PYTHONPATH=$PWD/:${PYTHONPATH:-}

if [ ! -f wav/wav.scp ]; then
    (cd wav && tar -zxvf wav.tar.gz)
fi

# model_dir includes model.pth.tar, asr_encoder.pth.tar, cmvn.ark, Qwen2-7B-Instruct
model_dir=$PWD/pretrained_models/FireRedASR2-LLM

accent=${1:-${ACCENT:-wuyu}}
data_jsonl=/n/work6/yiwang/chinavoices_challenge/chinavoices_challenge/reference_set/$accent/data.jsonl
data_root=/n/work6/yiwang/chinavoices_challenge
accent_data_dir=$PWD/data/$accent
wav_scp=$accent_data_dir/wav.scp
ref=$accent_data_dir/text

mkdir -p "$accent_data_dir"
python - "$data_jsonl" "$data_root" "$wav_scp" "$ref" "$accent" <<'PY'
import json
import os
import sys

jsonl_path, data_root, wav_scp_path, text_path, accent = sys.argv[1:]
num_utts = 0
missing = []

with open(jsonl_path, "r", encoding="utf-8") as fin, \
        open(wav_scp_path, "w", encoding="utf-8") as fwav, \
        open(text_path, "w", encoding="utf-8") as ftext:
    for line in fin:
        if not line.strip():
            continue
        item = json.loads(line)
        uttid = item["key"]
        wav_path = item["wav_path"]
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(data_root, wav_path)
        text = item.get("text", "")
        fwav.write(f"{uttid}\t{wav_path}\n")
        ftext.write(f"{uttid}\t{text}\n")
        num_utts += 1
        if not os.path.exists(wav_path):
            missing.append(wav_path)

if missing:
    print(f"Missing {len(missing)} wav files. First missing: {missing[0]}", file=sys.stderr)
    sys.exit(1)
print(f"Prepared {num_utts} {accent} utterances: {wav_scp_path}, {text_path}")
PY

wavs="--wav_scp $wav_scp"

out="out-$accent/llm-l-asr.txt"

decode_args="
--batch_size 1 --beam_size 3 --decode_max_len 0 --decode_min_len 0
--repetition_penalty 3.0 --llm_length_penalty 1.0 --temperature 1.0
"

mkdir -p $(dirname $out)
set -x


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
speech2text.py --asr_type "llm" --model_dir $model_dir $decode_args $wavs --output $out


wer.py --print_sentence_wer 1 --do_tn 1 --rm_special 1 --ref $ref --hyp $out > $out.wer 2>&1
tail -n8 $out.wer

#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"

echo "=== Qwen3 — val ==="
python scripts/eval_checkpoints.py \
    --train-config args/qwen3_coconut_ff_v3.yaml \
    --val-path ff_data/val_litereason.json \
    --eval-suffix val \
    --gpus "$GPUS"

echo "=== Qwen3 — test ==="
python scripts/eval_checkpoints.py \
    --train-config args/qwen3_coconut_ff_v3.yaml \
    --val-path ff_data/test_litereason.json \
    --eval-suffix test \
    --gpus "$GPUS" \
    --checkpoints 2,32

echo "=== Gemma3 — val ==="
python scripts/eval_checkpoints.py \
    --train-config args/gemma3_coconut_ff_v3.yaml \
    --val-path ff_data/val_litereason.json \
    --eval-suffix val \
    --gpus "$GPUS"

echo "=== Gemma3 — test ==="
python scripts/eval_checkpoints.py \
    --train-config args/gemma3_coconut_ff_v3.yaml \
    --val-path ff_data/test_litereason.json \
    --eval-suffix test \
    --gpus "$GPUS" \
    --checkpoints 4,32

echo "Done. Run: bash scripts/show_tables.sh"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPUS="${GPUS:-0,1,2,3}"

cd "$ROOT_DIR"

echo "=== Qwen3 — val ==="
python scripts/eval_checkpoints.py \
    --train-config args/legacy/qwen3_coconut_ff_v3.yaml \
    --val-path ff_data/val_litereason.json \
    --eval-suffix val \
    --gpus "$GPUS"

echo "=== Qwen3 — test ==="
python scripts/eval_checkpoints.py \
    --train-config args/legacy/qwen3_coconut_ff_v3.yaml \
    --val-path ff_data/test_litereason.json \
    --eval-suffix test \
    --gpus "$GPUS" \
    --checkpoints 2,32

echo "=== Gemma3 — val ==="
python scripts/eval_checkpoints.py \
    --train-config args/legacy/gemma3_coconut_ff_v3.yaml \
    --val-path ff_data/val_litereason.json \
    --eval-suffix val \
    --gpus "$GPUS"

echo "=== Gemma3 — test ==="
python scripts/eval_checkpoints.py \
    --train-config args/legacy/gemma3_coconut_ff_v3.yaml \
    --val-path ff_data/test_litereason.json \
    --eval-suffix test \
    --gpus "$GPUS" \
    --checkpoints 4,32

echo "Done. Run: bash scripts/legacy/show_tables.sh"

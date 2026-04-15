#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT_DIR/experiments/gsm_hard_qwen3_4b/train.yaml}"
GPUS="${GPUS:-0}"

cd "$ROOT_DIR"

CMD=(python scripts/eval_checkpoints.py --train-config "$CONFIG" --gpus "$GPUS")
if [[ -n "${CHECKPOINTS:-}" ]]; then
  CMD+=(--checkpoints "$CHECKPOINTS")
fi

exec "${CMD[@]}"

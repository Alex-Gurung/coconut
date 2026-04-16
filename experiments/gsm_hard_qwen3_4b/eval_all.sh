#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT_DIR/experiments/gsm_hard_qwen3_4b/train.yaml}"
GPUS="${GPUS:-0}"
TORCHRUN="${TORCHRUN:-torchrun}"

cd "$ROOT_DIR"

CMD=(
  python
  scripts/eval_checkpoints.py
  --train-config "$CONFIG"
  --gpus "$GPUS"
  --torchrun "$TORCHRUN"
)
if [[ -n "${CHECKPOINTS:-}" ]]; then
  CMD+=(--checkpoints "$CHECKPOINTS")
fi
if [[ -n "${VAL_PATH:-}" ]]; then
  CMD+=(--val-path "$VAL_PATH")
fi
if [[ -n "${EVAL_SUFFIX:-}" ]]; then
  CMD+=(--eval-suffix "$EVAL_SUFFIX")
fi
if [[ -n "${OUT:-}" ]]; then
  CMD+=(--out "$OUT")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  CMD+=(--skip-existing)
fi
if [[ "${LIST_ONLY:-0}" == "1" ]]; then
  CMD+=(--list-only)
fi

exec "${CMD[@]}"

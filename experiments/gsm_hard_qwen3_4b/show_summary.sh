#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT_DIR/experiments/gsm_hard_qwen3_4b/train.yaml}"
TARGET="${TARGET:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"

CMD=("$PYTHON_BIN" scripts/eval_summary_table.py)
if [[ -n "$TARGET" ]]; then
  CMD+=("$TARGET")
else
  CMD+=(--train-config "$CONFIG")
fi
if [[ -n "${EVAL_SUFFIX:-}" ]]; then
  CMD+=(--suffix "$EVAL_SUFFIX")
fi
if [[ -n "${SORT:-}" ]]; then
  CMD+=(--sort "$SORT")
fi
if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ "${REVERSE:-0}" == "1" ]]; then
  CMD+=(--reverse)
fi
if [[ "${REFRESH:-0}" == "1" ]]; then
  CMD+=(--refresh)
fi

exec "${CMD[@]}"

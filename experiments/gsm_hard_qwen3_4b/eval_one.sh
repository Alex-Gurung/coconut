#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="${CHECKPOINT:-${CKPT:-}}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT=<epoch>, for example: CHECKPOINT=4 bash experiments/gsm_hard_qwen3_4b/eval_one.sh" >&2
  exit 1
fi

cd "$ROOT_DIR"
export CHECKPOINTS="$CHECKPOINT"
exec bash "$ROOT_DIR/experiments/gsm_hard_qwen3_4b/eval_all.sh"

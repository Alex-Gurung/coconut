#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CONFIG="${1:-}"
GPUS="${2:-0,1,2,3}"

if [[ -z "${TRAIN_CONFIG}" ]]; then
  echo "Usage: $0 <train_config.yaml> [gpus_csv]"
  exit 1
fi

cd "$ROOT_DIR"

python scripts/eval_checkpoints.py \
  --train-config "${TRAIN_CONFIG}" \
  --gpus "${GPUS}"

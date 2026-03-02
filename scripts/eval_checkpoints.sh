#!/usr/bin/env bash
set -euo pipefail

TRAIN_CONFIG="${1:-}"
GPUS="${2:-0,1,2,3}"

if [[ -z "${TRAIN_CONFIG}" ]]; then
  echo "Usage: $0 <train_config.yaml> [gpus_csv]"
  exit 1
fi

/mnt/disk/coconut/new4/bin/python scripts/eval_checkpoints.py \
  --train-config "${TRAIN_CONFIG}" \
  --gpus "${GPUS}"

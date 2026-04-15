#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT_DIR/experiments/gsm_hard_qwen3_4b/train.yaml}"
NPROC="${NPROC_PER_NODE:-4}"

cd "$ROOT_DIR"
exec torchrun --nnodes 1 --nproc_per_node "$NPROC" run.py "$CONFIG"

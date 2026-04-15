#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT_DIR/experiments/gsm_hard_qwen3_4b/smoke.yaml}"

cd "$ROOT_DIR"
exec torchrun --nnodes 1 --nproc_per_node 1 run.py "$CONFIG"

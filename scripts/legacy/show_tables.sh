#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for f in \
    "$ROOT_DIR/checkpoints/qwen3-coconut-ff-v3/eval_summary-val.md" \
    "$ROOT_DIR/checkpoints/qwen3-coconut-ff-v3/eval_summary-test.md" \
    "$ROOT_DIR/checkpoints/gemma/gemma3-coconut-ff-v3/eval_summary-val.md" \
    "$ROOT_DIR/checkpoints/gemma/gemma3-coconut-ff-v3/eval_summary-test.md"
do
    if [ -f "$f" ]; then
        echo ""
        echo "=== $(basename "$(dirname "$f")") / $(basename "$f" .md) ==="
        cat "$f"
    else
        echo "Missing: $f"
    fi
done

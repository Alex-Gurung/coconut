#!/usr/bin/env bash
set -euo pipefail

for f in \
    checkpoints/qwen3-coconut-ff-v3/eval_summary-val.md \
    checkpoints/qwen3-coconut-ff-v3/eval_summary-test.md \
    checkpoints/gemma/gemma3-coconut-ff-v3/eval_summary-val.md \
    checkpoints/gemma/gemma3-coconut-ff-v3/eval_summary-test.md
do
    if [ -f "$f" ]; then
        echo ""
        echo "=== $(basename "$(dirname "$f")") / $(basename "$f" .md) ==="
        cat "$f"
    else
        echo "Missing: $f"
    fi
done

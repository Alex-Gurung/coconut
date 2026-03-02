#!/bin/bash
# Hyperparameter sweep: max_latent_stage × c_thought on Qwen3-1.7B
# Tests 4 configs: {s3,s10} × {c1,c2} with epochs_per_stage=2 fixed
#
# After training each config, evaluates ALL saved checkpoints (stage boundaries)
# to find the best epoch. Final analysis compares across all configs.
#
# Usage:
#   bash scripts/run_sweep.sh            # full sweep (4 GPUs)
#   bash scripts/run_sweep.sh --dry-run  # print commands without executing
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=4
CKPT_BASE=/mnt/disk/coconut/checkpoints

CONFIGS=(
  args/sweep_s3_c1.yaml
  args/sweep_s3_c2.yaml
  args/sweep_s10_c1.yaml
  args/sweep_s10_c2.yaml
)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "=== DRY RUN MODE ==="
fi

SWEEP_START=$(date +%s)

for config in "${CONFIGS[@]}"; do
  basename=$(basename "$config" .yaml)
  name=$(echo "$basename" | tr '_' '-')

  echo ""
  echo "========================================"
  echo "  Training: $config"
  echo "  Name: $name"
  echo "========================================"

  if $DRY_RUN; then
    echo "[dry-run] torchrun --nnodes 1 --nproc_per_node $NUM_GPUS run.py $config"
  else
    torchrun --nnodes 1 --nproc_per_node $NUM_GPUS run.py "$config"
  fi

  # Evaluate ALL checkpoints in this run's directory (directory mode).
  # Uses the training config as eval base so model_id/c_thought/max_latent_stage match.
  # scripts/eval/run_eval_n_times.py sets only_eval=True and resume=epoch automatically.
  ckpt_dir="$CKPT_BASE/$name"
  if [[ -d "$ckpt_dir" ]] && ! $DRY_RUN; then
    echo ""
    echo "  Evaluating all checkpoints in: $ckpt_dir"
    python scripts/eval/run_eval_n_times.py "$ckpt_dir" \
      --config-base "$config" \
      --num-gpus $NUM_GPUS
  elif $DRY_RUN; then
    echo "[dry-run] python scripts/eval/run_eval_n_times.py $ckpt_dir --config-base $config --num-gpus $NUM_GPUS"
  fi
done

SWEEP_END=$(date +%s)
SWEEP_ELAPSED=$((SWEEP_END - SWEEP_START))
echo ""
echo "========================================"
echo "  Sweep training + eval complete in ${SWEEP_ELAPSED}s"
echo "========================================"

# Run analysis to find the best config and checkpoint
echo ""
echo "  Running sweep analysis..."
if $DRY_RUN; then
  echo "[dry-run] python scripts/sweep_analysis.py $CKPT_BASE"
else
  python scripts/sweep_analysis.py "$CKPT_BASE"
fi

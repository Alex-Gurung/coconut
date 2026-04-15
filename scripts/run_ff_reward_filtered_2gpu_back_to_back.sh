#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk/coconut"
PYTHON_BIN="${PYTHON_BIN:-/tmp/.venv/bin/python}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"

GEMMA_CFG="${GEMMA_CFG:-args/gemma3_coconut_ff_reward_filtered_v1_2gpu.yaml}"
QWEN_CFG="${QWEN_CFG:-args/qwen3_coconut_ff_reward_filtered_v1_2gpu.yaml}"

QWEN_TRAIN="${QWEN_TRAIN:-$ROOT/ff_data/qwen_reward_filtered_train.json}"
QWEN_VAL="${QWEN_VAL:-$ROOT/ff_data/qwen_reward_filtered_val.json}"

QWEN_TRAIN_TRACE="${QWEN_TRAIN_TRACE:-/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/working/traces/ff_qwen3_4b_train_all.json}"
QWEN_VAL_TRACE="${QWEN_VAL_TRACE:-/mnt/disk/litereason_anon/litereason/experiments/flawed_fictions/working/traces/ff_qwen3_4b_val_all.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

gpu_count=$(awk -F',' '{print NF}' <<<"$GPUS")
if [[ "$gpu_count" -ne "$NPROC" ]]; then
  echo "GPUS=$GPUS implies $gpu_count visible GPUs, but NPROC=$NPROC" >&2
  exit 1
fi

cd "$ROOT"

if [[ ! -f "$QWEN_TRAIN" || ! -f "$QWEN_VAL" ]]; then
  echo "Qwen reward-filtered data missing. Building it now..."
  "$PYTHON_BIN" scripts/build_ff_reward_filtered_data.py \
    --trace "$QWEN_TRAIN_TRACE" "$QWEN_VAL_TRACE" \
    --train-out "$QWEN_TRAIN" \
    --val-out "$QWEN_VAL"
fi

echo "Gemma config: $GEMMA_CFG"
echo "Qwen config:  $QWEN_CFG"
echo "Using GPUs:   $GPUS"
echo "Using Python: $PYTHON_BIN"

run_train() {
  local cfg="$1"
  local label="$2"
  local run_meta
  run_meta=$("$PYTHON_BIN" - "$cfg" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["name"])
print(cfg["save_path"])
PY
)
  local run_name save_path save_dir
  run_name=$(sed -n '1p' <<<"$run_meta")
  save_path=$(sed -n '2p' <<<"$run_meta")
  save_dir="${save_path}/${run_name}"

  if [[ -d "$save_dir" ]]; then
    echo "Warning: $label save dir already exists: $save_dir" >&2
    echo "run.py will auto-resume if checkpoint_* files are present." >&2
  fi

  echo
  echo "=== Starting $label ==="
  CUDA_VISIBLE_DEVICES="$GPUS" \
    "$PYTHON_BIN" -m torch.distributed.run --nnodes 1 --nproc_per_node "$NPROC" run.py "$cfg"
}

run_train "$GEMMA_CFG" "Gemma 3 4B Coconut"
run_train "$QWEN_CFG" "Qwen 3 4B Coconut"

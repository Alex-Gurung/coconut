# Coconut (Clean Fork)

Minimal training/evaluation fork of COCONUT.

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare data
Training/eval data is JSON list format:
```json
[
  {"question": "...", "steps": ["...", "..."], "answer": "..."}
]
```

Default example paths used in configs:
- `ff_data/train.json`
- `ff_data/val.json`

### 3) Train
Copy and edit the example config:
```bash
cp args/example_train.yaml args/my_train.yaml
```

Run training:
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/my_train.yaml
```

### 4) Evaluate a checkpoint
Copy and edit the eval example:
```bash
cp args/example_eval.yaml args/my_eval.yaml
```

Run eval:
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/my_eval.yaml
```

## Project Layout
- `run.py`: canonical train/eval runner
- `evalrun.py`: thin compatibility wrapper to `run.py`
- `coconut.py`: Coconut model wrapper + generation logic
- `dataset.py`: tokenization and collator
- `args/`: experiment configs (copy from examples)
- `scripts/`: utilities (eval/data/sweeps/checks)
- `docs/`: deeper internal notes and runbooks

## Common Utility Commands
Batch eval across checkpoints:
```bash
python scripts/eval/run_eval_n_times.py /path/to/checkpoints --config-base args/my_eval.yaml --num-gpus 4
```

Combine eval outputs:
```bash
python scripts/eval/combine_evals.py /path/to/checkpoints
```

## Differences From Original
- For a clear change log against the original upstream repo (what changed, why, config args, and how to revert), see `docs/UPSTREAM_DIFF.md`.

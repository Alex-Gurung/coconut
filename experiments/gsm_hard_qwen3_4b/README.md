# GSM-Hard Qwen3-4B

Canonical workflow for training and evaluating Coconut on GSM-Hard with
`Qwen/Qwen3-4B-Instruct-2507`.

## Expected Data

This workflow expects Coconut-format JSON arrays at:

- `gsm_hard_data/qwen3_4b/train.json`
- `gsm_hard_data/qwen3_4b/val.json`

Each file should contain rows shaped like:

```json
[
  {
    "question": "...",
    "steps": ["...", "..."],
    "answer": "\\boxed{42}"
  }
]
```

## Files In This Folder

- `train.yaml`: main training config
- `smoke.yaml`: fast single-GPU smoke config
- `eval.yaml`: manual single-checkpoint eval template
- `train.sh`: wrapper for the main train config
- `smoke.sh`: wrapper for the smoke config
- `eval_all.sh`: wrapper around `scripts/eval_checkpoints.py`

## Train

Main run:

```bash
bash experiments/gsm_hard_qwen3_4b/train.sh
```

Override GPU count if needed:

```bash
NPROC_PER_NODE=2 bash experiments/gsm_hard_qwen3_4b/train.sh
```

## Smoke Test

```bash
bash experiments/gsm_hard_qwen3_4b/smoke.sh
```

## Evaluate Checkpoints

Evaluate every checkpoint from the main run:

```bash
GPUS=0,1,2,3 bash experiments/gsm_hard_qwen3_4b/eval_all.sh
```

Evaluate a specific checkpoint:

```bash
GPUS=0 CHECKPOINTS=4 bash experiments/gsm_hard_qwen3_4b/eval_all.sh
```

Results are written under:

```text
checkpoints/gsm_hard/<run-name>-eval-ckpt_XXX/eval_outputs.json
checkpoints/gsm_hard/<run-name>/eval_summary.json
checkpoints/gsm_hard/<run-name>/eval_summary.md
```

## Manual Single-Checkpoint Eval

`eval.yaml` is a template for direct `run.py` evals. Update:

- `name`
- `load_model_path`
- `resume`

Then run:

```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py experiments/gsm_hard_qwen3_4b/eval.yaml
```

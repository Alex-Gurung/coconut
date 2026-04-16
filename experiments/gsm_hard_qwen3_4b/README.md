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
- `UPSTREAM_NOTES.md`: why this workflow differs from original Coconut

## Train

Main run:

```bash
bash experiments/gsm_hard_qwen3_4b/train.sh
```

Override GPU count if needed:

```bash
NPROC_PER_NODE=2 bash experiments/gsm_hard_qwen3_4b/train.sh
```

The canonical config keeps the key behavior explicit:

- `use_chat_template: true`
- `answer_prefix: ""`
- `use_boxed_answer: true`
- `enable_gen_eval: false` during training

If you want the older post-chattemplate fork behavior instead, uncomment the
provided `answer_prefix: "In summary, "` line in the YAML configs.

With `save_every: 2` and `num_epochs: 20`, the expected checkpoint set is:

```text
checkpoint_2, checkpoint_4, checkpoint_6, ..., checkpoint_20
```

## Smoke Test

```bash
bash experiments/gsm_hard_qwen3_4b/smoke.sh
```

## Evaluate Checkpoints

List the checkpoints that would be evaluated without launching anything:

```bash
LIST_ONLY=1 bash experiments/gsm_hard_qwen3_4b/eval_all.sh
```

Evaluate every checkpoint from the main run:

```bash
GPUS=0,1,2,3 bash experiments/gsm_hard_qwen3_4b/eval_all.sh
```

Evaluate a specific checkpoint:

```bash
GPUS=0 CHECKPOINTS=4 bash experiments/gsm_hard_qwen3_4b/eval_all.sh
```

Re-run the sweep while reusing existing `eval_outputs.json` results:

```bash
GPUS=0,1,2,3 SKIP_EXISTING=1 bash experiments/gsm_hard_qwen3_4b/eval_all.sh
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
- `resume` (must match the checkpoint epoch number)

Then run:

```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py experiments/gsm_hard_qwen3_4b/eval.yaml
```

## Upstream Comparison

This path intentionally uses the cleaned-up canonical GSM-Hard defaults rather
than trying to be bit-for-bit identical to original Coconut or to every older
experiment in this fork.

For the short rationale behind the main deltas, see `UPSTREAM_NOTES.md`.

# Legacy Flawed Fictions Runbook

This document is now a legacy runbook for the Flawed Fictions workflow.

The primary workflow in this repo is:

- `experiments/gsm_hard_qwen3_4b/README.md`

Use this file only if you are intentionally working on the older Flawed Fictions
setup.

---

This runbook documents a strict, fail-fast pipeline for training and evaluating Coconut on Flawed Fictions data.

## Environment

Use your Python environment:

```bash
PYTHON_BIN=/mnt/disk/coconut/new4/bin/python
```

Install dependencies if needed:

```bash
$PYTHON_BIN -m pip install -r requirements.txt
```

## 1) Preprocess Data

This repo expects JSON arrays with:

```json
[
  {
    "question": "...",
    "steps": ["...", "..."],
    "answer": "\\boxed{Yes}"
  }
]
```

Your litereason trace files already match this schema. The command below merges Qwen+Gemma traces, drops invalid rows, deduplicates, and writes `ff_data/train.json` and `ff_data/val.json`.

```bash
$PYTHON_BIN scripts/legacy/ff_pipeline.py prepare-data
```

Quick sanity check:

```bash
$PYTHON_BIN scripts/legacy/ff_pipeline.py dataset-stats
```

## 2) Train

Use one of the Qwen3/Gemma3 configs (examples below).

Important:
- Change `name` for each new experiment. `run.py` auto-resumes if `save_path/name` already has checkpoints.
- This pipeline intentionally requires `flash_attention_2` and fails loudly if unavailable.

Tokenizer strict precheck:

```bash
$PYTHON_BIN scripts/legacy/ff_pipeline.py tokenizer-check --config args/legacy/qwen3_ff_coconut_strict_train.yaml
```

Example (4 GPUs):

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/legacy/qwen3_ff_coconut_strict_train.yaml
```

Full Qwen3 run (ff v1 settings):

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/legacy/qwen3_coconut_ff_v1_full.yaml
```

Smoke (2 epochs, 10% train data):

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/legacy/qwen3_coconut_ff_v1_smoke.yaml
torchrun --nnodes 1 --nproc_per_node 4 run.py args/legacy/gemma3_coconut_ff_v1_smoke.yaml
```

## 3) Evaluate

Use `args/legacy/qwen3_ff_coconut_strict_eval.yaml`:
- Set `load_model_path` to the checkpoint file.
- Set `resume` to the same checkpoint epoch (for matching latent stage).

Run eval:

```bash
torchrun --nnodes 1 --nproc_per_node 4 evalrun.py args/legacy/qwen3_ff_coconut_strict_eval.yaml
```

Outputs are saved to:

```text
/mnt/disk/coconut/checkpoints/<name>/eval_outputs.json
```

### Evaluate All Checkpoints (Accuracy, Parallel GPUs)

Use the helper script to evaluate every checkpoint and summarize accuracy:

```bash
PYTHON_BIN=/mnt/disk/coconut/new4/bin/python
$PYTHON_BIN scripts/eval_checkpoints.py \
  --train-config args/legacy/qwen3_coconut_ff_v1_full.yaml \
  --gpus 0,1,2,3
```

This writes:

```text
/mnt/disk/coconut/checkpoints/<run_name>/eval_summary.json
/mnt/disk/coconut/checkpoints/<run_name>/eval_summary.md
```

## 4) Multi-Run Eval (Optional)

`scripts/eval/run_eval_n_times.py` is configured to call `evalrun.py`.

```bash
$PYTHON_BIN scripts/eval/run_eval_n_times.py \
  /mnt/disk/coconut/checkpoints/qwen3-ff-coconut-v1/checkpoint_14 \
  5 \
  --config-base args/legacy/qwen3_ff_coconut_strict_eval.yaml \
  --num-gpus 4
```

Then combine:

```bash
$PYTHON_BIN scripts/eval/combine_evals.py /mnt/disk/coconut/checkpoints/qwen3-ff-coconut-v1
```

## Strictness And Compatibility Checks

These checks are now explicit:

- `run.py` and `evalrun.py` fail if tokenizer has no `eos_token`.
- `run.py` and `evalrun.py` fail if latent special tokens fail to register.
- `run.py` and `evalrun.py` fail if `latent_init_token` is not in vocab.
- `coconut.py` fails with a clear error if KV-cache hidden-state indexing is incompatible with the installed transformers behavior.

This keeps behavior deterministic and avoids silent fallback logic.

## Data / Curriculum Notes

`train_fraction` (optional) caps the train dataset size while keeping validation full.

## Scripted Helpers

Use `scripts/legacy/ff_pipeline.py` instead of ad-hoc inline commands:

```bash
$PYTHON_BIN scripts/legacy/ff_pipeline.py --help
```

Useful subcommands:
- `prepare-data`
- `dataset-stats`
- `tokenizer-check`
- `make-smoke-config`
- `make-train-config`
- `make-eval-config`
- `show-eval`

Example smoke flow:

```bash
$PYTHON_BIN scripts/legacy/ff_pipeline.py make-smoke-config --out args/legacy/generated/tmp_smoke_train.yaml
torchrun --nnodes 1 --nproc_per_node 1 run.py args/legacy/generated/tmp_smoke_train.yaml
```

Example full run + eval config generation:

```bash
$PYTHON_BIN scripts/legacy/ff_pipeline.py make-train-config --out args/legacy/generated/qwen3_ff_coconut_run.yaml
torchrun --nnodes 1 --nproc_per_node 4 run.py args/legacy/generated/qwen3_ff_coconut_run.yaml
$PYTHON_BIN scripts/legacy/ff_pipeline.py make-eval-config --train-config args/legacy/generated/qwen3_ff_coconut_run.yaml --out args/legacy/generated/qwen3_ff_coconut_run_eval.yaml
torchrun --nnodes 1 --nproc_per_node 4 evalrun.py args/legacy/generated/qwen3_ff_coconut_run_eval.yaml
$PYTHON_BIN scripts/legacy/ff_pipeline.py show-eval --eval-config args/legacy/generated/qwen3_ff_coconut_run_eval.yaml
```

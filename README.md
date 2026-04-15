# Coconut

Minimal training/evaluation fork of COCONUT with a canonical GSM-Hard workflow
for `Qwen/Qwen3-4B-Instruct-2507`.

## Primary Workflow

Start here:

- [`experiments/gsm_hard_qwen3_4b/README.md`](experiments/gsm_hard_qwen3_4b/README.md)
- [`experiments/gsm_hard_qwen3_4b/train.yaml`](experiments/gsm_hard_qwen3_4b/train.yaml)
- [`experiments/gsm_hard_qwen3_4b/smoke.yaml`](experiments/gsm_hard_qwen3_4b/smoke.yaml)
- [`experiments/gsm_hard_qwen3_4b/eval.yaml`](experiments/gsm_hard_qwen3_4b/eval.yaml)

Expected dataset paths for the primary workflow:

- `gsm_hard_data/qwen3_4b/train.json`
- `gsm_hard_data/qwen3_4b/val.json`

Main train command:

```bash
bash experiments/gsm_hard_qwen3_4b/train.sh
```

Smoke test:

```bash
bash experiments/gsm_hard_qwen3_4b/smoke.sh
```

Evaluate checkpoints from the main run:

```bash
GPUS=0,1,2,3 bash experiments/gsm_hard_qwen3_4b/eval_all.sh
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The runtime assumes you launch commands from the repo root.

## Project Layout

- `experiments/`: primary experiment folders with canonical configs and wrappers
- `run.py`: canonical train/eval runner
- `evalrun.py`: thin compatibility wrapper to `run.py`
- `coconut.py`: Coconut model wrapper + generation logic
- `dataset.py`: tokenization and collator
- `args/`: generic examples plus `args/legacy/` for older configs
- `scripts/`: primary helpers plus `scripts/legacy/` for older entrypoints
- `docs/`: indexes, upstream diff notes, `docs/legacy/`, and `docs/archive/`

## Legacy / Secondary Material

- For repo navigation, see `args/README.md`, `scripts/legacy/README.md`, and
  `docs/README.md`.
- Flawed Fictions workflow notes now live under `docs/legacy/` and
  `args/legacy/`.
- Checkpoint retention and recovery notes now live under `docs/archive/`.
- For upstream comparison notes, see `docs/UPSTREAM_DIFF.md`.
- For the exact upstream codebase, use the local `original_coconut` branch.

# Checkpoint Retention

This document is the pruning and archival companion to
`docs/CHECKPOINT_RECOVERY.md`.

`CHECKPOINT_RECOVERY.md` answers: "which raw checkpoints have exact public HF
recovery paths?"

This document answers two related questions:

- "if we do not want to keep every raw checkpoint locally, which ones are the
  most useful to retain on disk?"
- "which larger checkpoint superset is still worth archiving to HF before local
  pruning?"

The local keep set and the HF archive set are not the same:

- `balanced_local` is the smaller local-disk target after HF archival is
  verified
- `conservative_archive` is the larger HF archive target for future recovery and
  curve reconstruction

## Dry-Run Planner

Use the default local profile:

```bash
python scripts/checks/plan_checkpoint_prune.py
```

Inspect the larger HF archive profile:

```bash
python scripts/checks/plan_checkpoint_prune.py --profile conservative_archive
```

or machine-readable output:

```bash
python scripts/checks/plan_checkpoint_prune.py --json
```

The planner is dry-run only. It never deletes anything.

## Profiles

As of 2026-04-14:

- `balanced_local`: `43.6 GiB`
- `conservative_archive`: `101.8 GiB`
- delta between them: `58.2 GiB`

The current local disk also still contains non-selected checkpoints outside the
archive set. That is why the `balanced_local` dry run shows more reclaimable
space than `101.8 - 43.6 GiB`.

## Retention Manifest

The machine-readable keep sets live in:

```text
scripts/checks/checkpoint_retention_manifest.json
```

The profile memberships were chosen using three rules:

- preserve checkpoints explicitly used in downstream eval or row-recompute work
- preserve checkpoints that anchor major transitions in the recorded validation
  or test curves
- preserve final/exported checkpoints that already back local aliases or HF
  exports

## Conservative Archive Set

These entries are the checkpoints worth preserving somewhere. The
`balanced_local` subset is identified inline.

### `checkpoints/qwen3-coconut-ff-v3`

Keep:

- `checkpoint_24` (`balanced_local`)
- `checkpoint_32`

Reasoning:

- These are the only raw checkpoints still present locally.
- Both appear in the recorded standard FF val/test summaries.
- `checkpoint_32` is the exact HF-backed export.

### `checkpoints/gemma/gemma3-coconut-ff-v3`

Keep:

- `checkpoint_32`

Reasoning:

- It is the only raw checkpoint still present locally.
- It is the exact HF-backed export.

### `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu`

Keep:

- `checkpoint_20` (`balanced_local`)
- `checkpoint_22`
- `checkpoint_24` (`balanced_local`)
- `checkpoint_26`
- `checkpoint_32` (`balanced_local`)

Reasoning:

- `checkpoint_20` was used in the March 31 row-recompute pipeline and a
  successful 4-shard rerun.
- `checkpoint_22` was explicitly tested on March 27 and preserves the remembered
  mid-run checkpoint.
- `checkpoint_24` is the first major jump checkpoint in the recorded val/test
  results.
- `checkpoint_26` is a high-performing late-stage checkpoint with explicit test
  outputs.
- `checkpoint_32` is the final late-stage checkpoint and one of the best
  recorded late checkpoints.

Alternative:

- Swap in `checkpoint_16` if you prefer a better pre-jump curve anchor over
  retaining the explicitly tested `checkpoint_22`.

Important note:

- The validation sweep also includes `checkpoint_8` and `checkpoint_12`, but
  those raw checkpoint files are already gone from the local run directory.

### `checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu`

Keep:

- `checkpoint_20`
- `checkpoint_24`
- `checkpoint_28`
- `checkpoint_32` (`balanced_local`)

Reasoning:

- This run only has four raw checkpoints on disk.
- `checkpoint_32` is the only one with explicit in-repo eval summaries.
- Since the run is already small, the recommended retention set is simply all
  four checkpoints.

### `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard`

Keep:

- `checkpoint_4` (`balanced_local`)
- `checkpoint_8`
- `checkpoint_12`
- `checkpoint_24` (`balanced_local`)
- `checkpoint_32` (`balanced_local`)

Reasoning:

- `checkpoint_4` is the best checkpoint in the recorded GSM-Hard validation
  sweep.
- `checkpoint_8` and `checkpoint_12` preserve the early declining part of the
  curve before collapse.
- `checkpoint_24` is the best late-stage non-zero checkpoint.
- `checkpoint_32` is the final checkpoint used by downstream row-recompute
  scripts and by the local materialized standard export.

## Current HF Placement Check

Re-run the current namespace report with:

```bash
/mnt/disk/litereason_anon/.venv/bin/python scripts/checks/report_agurung_namespace.py
```

As checked on 2026-04-14:

- Existing public Coconut repos already cover the standard FF exports:
  - `agurung/coconut-qwen3-4b-ff`
  - `agurung/coconut-gemma-3-4b-ff`
  - `agurung/qwen-coconut-ff-v2`
- Those standard Coconut repos currently expose only `main` branches, except
  `agurung/qwen-coconut-ff-v2`, which also has tag `checkpoint-13`.
- There are no dedicated public Coconut repos yet for:
  - Qwen3 FF reward-filtered
  - Gemma 3 4B FF reward-filtered
  - Gemma 3 1B GSM-Hard Coconut

## Suggested Upload Targets

If you want public homes for the retained non-standard runs, the cleanest repo
targets are:

- `agurung/coconut-qwen3-4b-ff-reward-filtered`
- `agurung/coconut-gemma-3-4b-ff-reward-filtered`
- `agurung/coconut-gemma-3-1b-gsm-hard`

These do not currently exist in the public namespace scan.

Using the existing standard FF Coconut repos as extra branches is technically
possible, but it would mix standard and reward-filtered exports under the same
repo identity. Separate repos are cleaner for later recovery and for model cards.

## Faithful HF Publishing

The machine-readable archive plan lives in:

```text
scripts/checks/checkpoint_publish_manifest.json
```

The executor for that manifest is:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/train/publish_coconut_checkpoints.py \
  --profile conservative_archive \
  --execute
```

That executor uses the faithful Coconut export path, uploads branch/tagged HF
snapshots, and verifies the returned immutable commit SHA before considering the
upload complete.

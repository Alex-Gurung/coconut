# Checkpoint Retention

This document is the pruning and archival companion to
`docs/archive/CHECKPOINT_RECOVERY.md`.

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

This document now makes the keep logic explicit. For each run, the policy tries
to cover four core buckets:

- `used_elsewhere`
- `best_eval_acc`
- `best_eval_loss_or_proxy`
- `last`

Any extra checkpoints kept only in `conservative_archive` are marked as
`extras`.

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

As of 2026-04-15:

- `balanced_local`: `43.6 GiB`
- `conservative_archive`: `101.8 GiB`
- delta between them: `58.2 GiB`

The current local disk also still contains non-selected checkpoints outside the
archive set. That is why the `balanced_local` dry run shows more reclaimable
space than `101.8 - 43.6 GiB`.

After the 2026-04-15 archival pass, every checkpoint in
`conservative_archive` has an exact recorded HF recovery path in
`docs/archive/CHECKPOINT_RECOVERY.md`. The only local-only checkpoints left are
the `73.0 GiB` outside that archive set.

## Retention Manifest

The machine-readable keep sets live in:

```text
scripts/checks/checkpoint_retention_manifest.json
```

The profile memberships are encoded in the manifest with explicit bucket labels.
The high-level rule is:

- cover `used_elsewhere`, `best_eval_acc`, `best_eval_loss_or_proxy`, and
  `last` whenever the local evidence supports those buckets
- use a proxy when comparable `eval_loss` is not durably recorded for the run
- keep additional curve anchors or explicitly tested checkpoints as
  `conservative_archive`-only `extras`

For these Coconut runs, a clean cross-run `eval_loss` table is usually not
available. In practice, `best_eval_loss_or_proxy` means one of:

- lowest `avg_completion_tokens` on a meaningful plateau
- first major jump checkpoint in a recorded sweep
- best late-stage non-zero checkpoint
- the only checkpoint with explicit in-repo eval outputs

## Conservative Archive Set

These entries are the checkpoints worth preserving somewhere. The
`balanced_local` subset is identified inline, and the bucket tags explain why
each checkpoint survives.

### `checkpoints/qwen3-coconut-ff-v3`

Keep:

- `checkpoint_24` (`balanced_local`) -> `best_eval_acc`,
  `best_eval_loss_or_proxy`
- `checkpoint_32` -> `last`

Reasoning:

- These are the only raw checkpoints still present locally.
- The late-stage standard FF summaries effectively reduce this run to
  "best/tie plus last."
- `checkpoint_32` is also the exact HF-backed export.

### `checkpoints/gemma/gemma3-coconut-ff-v3`

Keep:

- `checkpoint_32` -> `best_eval_acc`, `best_eval_loss_or_proxy`, `last`

Reasoning:

- It is the only raw checkpoint still present locally.
- It is the exact HF-backed export.
- In practice it carries every core bucket that still has local evidence.

### `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu`

Keep:

- `checkpoint_20` (`balanced_local`) -> `used_elsewhere`
- `checkpoint_22` -> `extras`
- `checkpoint_24` (`balanced_local`) -> `best_eval_loss_or_proxy`
- `checkpoint_26` -> `extras`
- `checkpoint_32` (`balanced_local`) -> `best_eval_acc`, `last`

Reasoning:

- `checkpoint_20` was used in the March 31 row-recompute pipeline and a
  successful 4-shard rerun.
- `checkpoint_24` is the first major jump checkpoint and the first checkpoint on
  the lowest `avg_completion_tokens` plateau.
- `checkpoint_32` is the final late-stage checkpoint on the tied-best
  validation plateau.
- `checkpoint_22` and `checkpoint_26` are conservative extras kept for explicit
  historical test coverage and curve reconstruction.

Alternative:

- Swap in `checkpoint_16` if you prefer a better pre-jump curve anchor over
  retaining the explicitly tested `checkpoint_22`.

Important note:

- The validation sweep also includes `checkpoint_8` and `checkpoint_12`, but
  those raw checkpoint files are already gone from the local run directory.
- `balanced_local` is already the minimal three-checkpoint set that covers
  `used_elsewhere`, proxy-loss, and best-acc-plus-last for this run.

### `checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu`

Keep:

- `checkpoint_20` -> `extras`
- `checkpoint_24` -> `extras`
- `checkpoint_28` -> `extras`
- `checkpoint_32` (`balanced_local`) -> `best_eval_acc`,
  `best_eval_loss_or_proxy`, `last`

Reasoning:

- This run only has four raw checkpoints on disk.
- `checkpoint_32` is the only checkpoint with explicit in-repo eval summaries,
  so it carries the core buckets.
- The earlier checkpoints remain `conservative_archive` extras because the run
  is small and sparsely instrumented.

### `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard`

Keep:

- `checkpoint_4` (`balanced_local`) -> `best_eval_acc`
- `checkpoint_8` -> `extras`
- `checkpoint_12` -> `extras`
- `checkpoint_24` (`balanced_local`) -> `best_eval_loss_or_proxy`
- `checkpoint_32` (`balanced_local`) -> `used_elsewhere`, `last`

Reasoning:

- `checkpoint_4` is the best checkpoint in the recorded GSM-Hard validation
  sweep.
- `checkpoint_24` is the best late-stage non-zero checkpoint and acts as the
  loss/quality proxy for the late run.
- `checkpoint_32` is the final checkpoint used by downstream row-recompute
  scripts and by the local materialized standard export.
- `checkpoint_8` and `checkpoint_12` remain conservative extras that preserve
  the early declining curve before collapse.

## Current HF Placement Check

Re-run the current namespace report with:

```bash
/mnt/disk/litereason_anon/.venv/bin/python scripts/checks/report_agurung_namespace.py
```

The non-standard archive repos now exist publicly:

- `agurung/coconut-qwen3-4b-ff-reward-filtered`
- `agurung/coconut-gemma-3-4b-ff-reward-filtered`
- `agurung/coconut-gemma-3-1b-gsm-hard`

Use the namespace report for live refs, and use
`docs/archive/CHECKPOINT_RECOVERY.md` as the source of truth for "safe to
delete" decisions. A public branch or tag is not enough by itself; only an
exact recorded round-trip verification counts.

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

That executor uses the faithful Coconut export path for any unpublished entry.
Once an entry has an exact verified commit SHA recorded in
`checkpoint_publish_manifest.json`, reruns collapse to `create_refs_only` so the
manifest simply reasserts branch/tag refs against that immutable revision.

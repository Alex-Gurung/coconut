# Upstream Differences (This Repo vs Original COCONUT)

Compared against: `/mnt/disk/old_coconut/coconut`

## At a glance
- Shared files changed: `run.py`, `dataset.py`, `coconut.py`, `README.md`
- New tracked files in this repo: many configs/scripts/docs for Flawed Fictions and eval tooling
- Root is intentionally cleaner: most utilities moved to `scripts/`, docs moved to `docs/`

Core diff size (shared files):
- `run.py`: `+409 / -101`
- `dataset.py`: `+133 / -22`
- `coconut.py`: `+124 / -23`
- `README.md`: rewritten for quickstart

## 1) Behavioral changes in `run.py`

### What changed
- `run.py` is now the canonical entrypoint for both train and eval.
- `evalrun.py` is now a thin wrapper that just calls `run.main()`.
- Added optional runtime features:
  - DDP toggle (`use_ddp`) in addition to FSDP
  - `torch_compile` toggle
  - Liger kernel attempt at startup
- Added stronger validation/guardrails:
  - tokenizer/eos/special-token checks
  - optional smoke single-GPU guard
- Added eval controls:
  - `enable_gen_eval` gate (skip slow generation eval when false)
  - `max_new_tokens` override from config
  - `max_eval_samples` support for quick eval subsets
  - eval output dedup by `idx` in distributed eval
- Added checkpoint/eval operational controls:
  - `save_every`
  - `train_fraction`
  - non-default optional config logging at startup

### Revert to original behavior
Use config values that mimic upstream defaults:
- `use_ddp: false`
- `torch_compile: false`
- `enable_gen_eval: true` (if you want per-epoch generation eval like old behavior)
- `use_chat_template: false`
- `save_every: 1`
- `train_fraction: null`
- `max_new_tokens: null` (uses heuristic fallback)
- `max_eval_samples: null` (full eval)

## 2) Behavioral changes in `dataset.py`

### What changed
- Added `use_chat_template` path in `get_dataset(...)`.
- If enabled, tokenization uses `tokenizer.apply_chat_template(...)`.
- Collator now ensures `token_type_ids` exists and shape-checks it.
- Current answer prefix in this fork is `"In summary, "` (instead of old `"### "` formatting).

### Revert to original behavior
- Set `use_chat_template: false`.
- If you need exact old formatting, switch answer tokenization back from `"In summary, "` to `"### "` in `dataset.py` (this is a code change, not just config).

## 3) Behavioral changes in `coconut.py`

### What changed
- Added cache compatibility for newer transformers cache API (`Cache` handling).
- Added optional sampling support in generation:
  - `do_sample`, `temperature`, `top_p`, `top_k`
- Added optional `token_type_ids` threading through model calls.
- Added explicit hidden-state/cache index mismatch guard.

### Revert to original behavior
- Keep eval sampling disabled:
  - `eval_do_sample: false`
- For strict old decoding behavior, generation remains greedy when sampling is off.
- Full code-level reversion would require restoring old `coconut.py` from upstream.

## 4) New config args in this fork (most important)

These are the key args not present in original upstream configs:
- `use_chat_template`
- `enable_gen_eval`
- `use_boxed_answer`
- `max_new_tokens`
- `max_eval_samples`
- `eval_do_sample`
- `eval_temperature`
- `eval_top_p`
- `eval_top_k`
- `use_ddp`
- `torch_compile`
- `save_every`
- `latent_init_token`
- `train_fraction`
- `smoke_single_gpu`

Reference examples:
- `args/example_train.yaml`
- `args/example_eval.yaml`

## 5) Repo structure differences

### What changed
- Utilities moved from root to grouped folders:
  - `scripts/checks/`, `scripts/data/`, `scripts/eval/`, `scripts/sweeps/`, `scripts/train/`
- Internal docs moved to `docs/`
- Tests moved to `tests/`

### Revert to original layout
- Move those files back to repo root and update path references in scripts/docs.

## 6) Flawed Fictions / Qwen-specific additions

### What changed
- Added Qwen3 + Flawed Fictions experiment configs in `args/legacy/`.
- Added helper scripts for batched checkpoint eval and sweep analysis.

### Revert
- Remove those extra configs/scripts and keep only upstream-compatible files.

## 7) Direct answer to your question: chat templates

Yes. This fork supports chat-template tokenization via:
- config: `use_chat_template: true|false`
- code path: `dataset.py -> get_dataset(..., use_chat_template=...)`

If `use_chat_template: false`, tokenization follows the non-chat path.

## 8) If you want “closest to upstream” starting config

Start from `args/example_train.yaml` and set:
- `use_chat_template: false`
- `enable_gen_eval: true`
- `eval_do_sample: false`
- `use_ddp: false`
- `torch_compile: false`
- `train_fraction: null`
- `max_eval_samples: null`
- `max_new_tokens: null`

Then optionally restore old answer prefix behavior in `dataset.py` (`###`).

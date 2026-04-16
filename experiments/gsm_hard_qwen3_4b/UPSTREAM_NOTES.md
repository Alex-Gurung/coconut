# GSM-Hard Notes Vs Original Coconut

Compared against the local `original_coconut` branch.

This experiment folder defines the current canonical GSM-Hard behavior for
`Qwen/Qwen3-4B-Instruct-2507`. The main deltas from upstream are:

- `use_chat_template: true`
  Reason: the target model is instruct-tuned, so the tokenizer's chat template
  is the intended prompt format for this workflow.
- `answer_prefix: ""`
  Reason: the prepared GSM-Hard data stores boxed answers directly, so the
  canonical path trains against the answer text itself instead of prepending the
  older fork-specific `"In summary, "` string.
- `use_boxed_answer: true`
  Reason: GSM-Hard answers are boxed/numeric in the prepared dataset, so eval
  extracts the boxed final answer before comparing correctness.
- `resume` must match the checkpoint epoch during eval
  Reason: Coconut's latent stage schedule depends on epoch number, so
  checkpoint evaluation has to preserve that mapping. `eval_all.sh` and
  `scripts/eval_checkpoints.py` do this automatically.
- `enable_gen_eval: false` during training
  Reason: the intended workflow is to keep training cheaper, then run a single
  consistent checkpoint sweep after training.
- `save_every: 2`
  Reason: the canonical run is expected to produce a regular checkpoint set that
  can be evaluated exhaustively after training.

Operational deltas that matter less scientifically:

- `use_ddp: true` is the default wrapper choice for this path.
- `flash_attention_2`, gradient checkpointing, and optional Liger hooks are
  runtime optimizations rather than Coconut algorithm changes.
- `disable_kv_cache` exists for model families that need a full-recompute path,
  but the canonical Qwen3 GSM-Hard configs leave it off.
- The older `"In summary, "` answer prefix is still shown as a commented option
  in the canonical YAMLs for compatibility with older fork runs.

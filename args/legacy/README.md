# Legacy Args

These configs are preserved for older research runs, comparisons, and recovery.

They are not the default entry point for this repo.

Use the primary GSM-Hard workflow instead:

- `experiments/gsm_hard_qwen3_4b/train.yaml`
- `experiments/gsm_hard_qwen3_4b/smoke.yaml`
- `experiments/gsm_hard_qwen3_4b/eval.yaml`

The files here include:

- older GSM / Qwen / ProsQA / ProntoQA configs
- Flawed Fictions training and eval configs
- sweep configs
- temporary smoke configs retained from earlier runs

Compatibility notes:

- chat-template legacy configs pin `answer_prefix: "In summary, "` where they
  were historically using the fork-specific post-chattemplate format
- non-chat legacy configs pin `answer_prefix: "### "` where they are intended
  to preserve the older upstream-style formatting
- legacy eval and `save_only_improve` configs pin `enable_gen_eval: true` so
  they still run generation accuracy instead of inheriting the canonical train
  default

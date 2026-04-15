# Args Directory

This directory contains legacy and secondary config files accumulated across
multiple experiments.

Primary workflow:

- use `experiments/gsm_hard_qwen3_4b/train.yaml`
- use `experiments/gsm_hard_qwen3_4b/smoke.yaml`
- use `experiments/gsm_hard_qwen3_4b/eval.yaml`

Notes:

- `example_train.yaml` and `example_eval.yaml` are generic examples, not the
  main recommended path
- many `qwen3_*ff*`, `gemma3_*ff*`, and sweep configs are Flawed Fictions or
  other research-specific variants
- older `gsm_*`, `qwen_*`, `prosqa_*`, and `prontoqa_*` configs are retained
  for history and comparison, not as the default entry point

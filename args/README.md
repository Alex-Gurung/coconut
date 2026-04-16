# Args Directory

This directory contains generic examples plus a `legacy/` folder with older
research configs accumulated across multiple experiments.

Primary workflow:

- use `experiments/gsm_hard_qwen3_4b/train.yaml`
- use `experiments/gsm_hard_qwen3_4b/smoke.yaml`
- use `experiments/gsm_hard_qwen3_4b/eval.yaml`

Notes:

- `example_train.yaml` and `example_eval.yaml` are generic examples, not the
  main recommended path
- older research configs now live under `args/legacy/`
- legacy chat-template configs pin `answer_prefix: "In summary, "` explicitly
  where needed so they keep their historical behavior
- legacy non-chat configs pin `answer_prefix: "### "` where they are meant to
  stay closer to pre-chattemplate/upstream formatting
- legacy eval and `save_only_improve` configs pin `enable_gen_eval: true`
  explicitly so they do not silently skip generation accuracy

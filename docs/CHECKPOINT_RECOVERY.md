# Checkpoint Recovery

This document tracks which local Coconut checkpoints have been verified as safe to
replace with a Hugging Face pull, and which ones still require a local copy.

## Full Audit As Of 2026-04-14

All local `checkpoints/**/checkpoint_*` files were audited with
`scripts/checks/audit_coconut_checkpoints.py`.

- Local checkpoint files audited: `30`
- Exact round-trip verified against an HF commit SHA: `3`
- Already replaced locally by an HF offload placeholder: `1`
- No exact HF recovery path found yet: `26`

The audit scanned every public `agurung` model repo/ref that exposes
`latent_metadata.json` and matched them against the exact local checkpoint paths in
this workspace.

One public repo, `agurung/coconut-qwen2.5-7b`, exists but does not expose
`latent_metadata.json`, so it is not an exact recovery source for any local
checkpoint listed here.

## Safe To Delete After You Decide To Prune

These checkpoints were round-trip verified against exact HF commit SHAs:

- raw `latent_checkpoint.pt` downloaded from HF matches the local checkpoint by SHA256
- every exported base-model tensor matches the local checkpoint across all layers
- the tokenizer in the HF snapshot contains the Coconut latent tokens

### `checkpoints/qwen3-coconut-ff-v3/checkpoint_32`

- HF repo: `agurung/coconut-qwen3-4b-ff`
- Exact revision: `92248e30dd55c021b5c3950ad9c386f861b104f2`
- Local checkpoint SHA256:
  `16ab280db425b8c4dd24635a230b661260cc46899ff2bf27e7ae4aac8ffb3a69`
- Tokenizer:
  `<|start-latent|>` -> `151669`
  `<|end-latent|>` -> `151670`
  `<|latent|>` -> `151671`
- Verification summary:
  `399` base tensors matched exactly
  raw `latent_checkpoint.pt` matched exactly
  HF snapshot includes `latent_metadata.json`

Pull the exact verified snapshot:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="agurung/coconut-qwen3-4b-ff",
    revision="92248e30dd55c021b5c3950ad9c386f861b104f2",
    local_dir="restore_qwen3_coconut_ff_ckpt32",
)
```

The exact raw checkpoint will be at:

```text
restore_qwen3_coconut_ff_ckpt32/latent_checkpoint.pt
```

### `checkpoints/gemma/gemma3-coconut-ff-v3/checkpoint_32`

- HF repo: `agurung/coconut-gemma-3-4b-ff`
- Exact revision: `f59d39cf36b126140a2b775ef7d82643f8bceb17`
- Local checkpoint SHA256:
  `6138146c8f34f8885891ca0949b7a265195e2c9a6049eaf681b6a6a8604b208e`
- Tokenizer:
  `<|start-latent|>` -> `262145`
  `<|end-latent|>` -> `262146`
  `<|latent|>` -> `262147`
- Verification summary:
  `884` base tensors matched exactly
  raw `latent_checkpoint.pt` matched exactly
  HF snapshot includes `latent_metadata.json`

Pull the exact verified snapshot:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="agurung/coconut-gemma-3-4b-ff",
    revision="f59d39cf36b126140a2b775ef7d82643f8bceb17",
    local_dir="restore_gemma3_coconut_ff_ckpt32",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_coconut_ff_ckpt32/latent_checkpoint.pt
```

### `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_4`

- HF repo: `agurung/coconut-gemma-3-1b-gsm-hard`
- Exact revision: `4c2bdcaa2934616d2f0c7a0f1d0f5ba611ec5db5`
- HF refs observed after publish: `checkpoint_4`, `checkpoint-4`
- Local checkpoint SHA256:
  `2ce98967be3d37e57faa45541ecff5af69e24a9bf187c4ce3b30cba346d77fbe`
- Tokenizer:
  `<|start-latent|>` -> `262145`
  `<|end-latent|>` -> `262146`
  `<|latent|>` -> `262147`
- Verification summary:
  `341` base tensors matched exactly
  raw `latent_checkpoint.pt` matched exactly
  HF snapshot includes `latent_metadata.json`

Pull the exact verified snapshot:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="agurung/coconut-gemma-3-1b-gsm-hard",
    revision="4c2bdcaa2934616d2f0c7a0f1d0f5ba611ec5db5",
    local_dir="restore_gemma3_1b_coconut_gsm_hard_ckpt4",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_1b_coconut_gsm_hard_ckpt4/latent_checkpoint.pt
```

## Already Offloaded Earlier

### `checkpoints/qwen-coconut-ff-v2/checkpoint_13`

- This local checkpoint is already a placeholder, not a real `.pt` file.
- HF repo: `agurung/qwen-coconut-ff-v2`
- Exact revision: `0ce728a93420ec5cd2380cbed16f3e7acbf1a292`
- HF refs observed in the audit: `main`, `checkpoint-13`
- HF repo contains `latent_checkpoint.pt`

Pull it with:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="agurung/qwen-coconut-ff-v2",
    revision="0ce728a93420ec5cd2380cbed16f3e7acbf1a292",
    local_dir="restore_qwen25_coconut_ff_v2",
)
```

The raw checkpoint is stored in that repo as `latent_checkpoint.pt`.

## Not Safe To Delete Yet

No public `agurung` model repo/ref with `latent_metadata.json` points to any of
these exact local checkpoint paths, so they do not yet have an exact HF recovery
path recorded here:

- `checkpoints/qwen3-coconut-ff-v3`: `checkpoint_24`
- `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu`: `checkpoint_2`, `checkpoint_4`, `checkpoint_6`, `checkpoint_10`, `checkpoint_14`, `checkpoint_16`, `checkpoint_18`, `checkpoint_20`, `checkpoint_22`, `checkpoint_24`, `checkpoint_26`, `checkpoint_28`, `checkpoint_30`, `checkpoint_32`
- `checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu`: `checkpoint_20`, `checkpoint_24`, `checkpoint_28`, `checkpoint_32`
- `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard`: `checkpoint_8`, `checkpoint_12`, `checkpoint_16`, `checkpoint_20`, `checkpoint_24`, `checkpoint_28`, `checkpoint_32`

## Re-Verification

Use the scripted verifier after any future upload:

```bash
python scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/qwen3-coconut-ff-v3/checkpoint_32 \
  agurung/coconut-qwen3-4b-ff \
  92248e30dd55c021b5c3950ad9c386f861b104f2
```

and:

```bash
python scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gemma/gemma3-coconut-ff-v3/checkpoint_32 \
  agurung/coconut-gemma-3-4b-ff \
  f59d39cf36b126140a2b775ef7d82643f8bceb17
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_4 \
  agurung/coconut-gemma-3-1b-gsm-hard \
  4c2bdcaa2934616d2f0c7a0f1d0f5ba611ec5db5
```

To audit every local checkpoint path and round-trip verify any exact HF match that
is found:

```bash
python scripts/checks/audit_coconut_checkpoints.py \
  --verify-matches \
  --download-root /tmp/coconut_checkpoint_audit
```

The audit requires an environment with `huggingface_hub` plus a recent
`transformers` build that recognizes `qwen3` and `gemma3`.

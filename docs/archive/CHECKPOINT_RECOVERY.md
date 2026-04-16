# Checkpoint Recovery

Archive note:

- This document is archival/reference material, not part of the primary
  GSM-Hard workflow.

This document tracks which local Coconut checkpoints have been verified as safe to
replace with a Hugging Face pull, and which ones still require a local copy.

## Full Audit As Of 2026-04-15

All local `checkpoints/**/checkpoint_*` files were audited with
`scripts/checks/audit_coconut_checkpoints.py`.

- Local checkpoint files audited: `30`
- Exact round-trip verified against an HF commit SHA: `17`
- Already replaced locally by an HF offload placeholder: `1`
- No exact HF recovery path found yet: `12`

The audit scanned every public `agurung` model repo/ref that exposes
`latent_metadata.json` and matched them against the exact local checkpoint paths in
this workspace.

One public repo, `agurung/coconut-qwen2.5-7b`, exists but does not expose
`latent_metadata.json`, so it is not an exact recovery source for any local
checkpoint listed here.

Every checkpoint selected by
`scripts/checks/checkpoint_publish_manifest.json` for profile
`conservative_archive` now has an exact HF recovery path recorded below.

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

### `checkpoints/qwen3-coconut-ff-v3/checkpoint_24`

- HF repo: `agurung/coconut-qwen3-4b-ff`
- Exact revision: `ab2ee7618ea3d87be00d7f1de9328a9144abb48e`
- HF refs observed after republish: `checkpoint_24`, `checkpoint-24`
- Local checkpoint SHA256:
  `0602b2430cbc8b0a682ef4476fa43b08a773375eda7a3dd682674e0c2d6de216`
- Tokenizer:
  `<|start-latent|>` -> `151669`
  `<|end-latent|>` -> `151670`
  `<|latent|>` -> `151671`
- Verification summary:
  `399` base tensors matched exactly
  raw `latent_checkpoint.pt` matched exactly
  stale `model.safetensors` was removed before republish
  HF snapshot includes `latent_metadata.json`

Pull the exact verified snapshot:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="agurung/coconut-qwen3-4b-ff",
    revision="ab2ee7618ea3d87be00d7f1de9328a9144abb48e",
    local_dir="restore_qwen3_coconut_ff_ckpt24",
)
```

The exact raw checkpoint will be at:

```text
restore_qwen3_coconut_ff_ckpt24/latent_checkpoint.pt
```

### `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_20`

- HF repo: `agurung/coconut-qwen3-4b-ff-reward-filtered`
- Exact revision: `7b67e6f88823ece6ed7be74d1847719044f98efc`
- HF refs observed after publish: `checkpoint_20`, `checkpoint-20`
- Local checkpoint SHA256:
  `3c183f292141a86a2b01d40f3d61d52ff6409e97a9b7b51ed71f09284f72e242`
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
    repo_id="agurung/coconut-qwen3-4b-ff-reward-filtered",
    revision="7b67e6f88823ece6ed7be74d1847719044f98efc",
    local_dir="restore_qwen3_coconut_ff_reward_filtered_ckpt20",
)
```

The exact raw checkpoint will be at:

```text
restore_qwen3_coconut_ff_reward_filtered_ckpt20/latent_checkpoint.pt
```

### `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_22`

- HF repo: `agurung/coconut-qwen3-4b-ff-reward-filtered`
- Exact revision: `90d513c9b995ddb309a9d5b8735a02c6008b49a8`
- HF refs observed after publish: `checkpoint_22`, `checkpoint-22`
- Local checkpoint SHA256:
  `878218a1bc456ca54ba1b1e52a26e5cb6bafcaa0e18df5653c652bf9eb657c3e`
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
    repo_id="agurung/coconut-qwen3-4b-ff-reward-filtered",
    revision="90d513c9b995ddb309a9d5b8735a02c6008b49a8",
    local_dir="restore_qwen3_coconut_ff_reward_filtered_ckpt22",
)
```

The exact raw checkpoint will be at:

```text
restore_qwen3_coconut_ff_reward_filtered_ckpt22/latent_checkpoint.pt
```

### `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_24`

- HF repo: `agurung/coconut-qwen3-4b-ff-reward-filtered`
- Exact revision: `06140ea0d09d4cebbbe0334c3cfa0211ff368318`
- HF refs observed after publish: `checkpoint_24`, `checkpoint-24`
- Local checkpoint SHA256:
  `45fe98d5b55cb5bb47ea2b74a0b0c16fc5b797b0f1ce4790c9855d41b754d8b5`
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
    repo_id="agurung/coconut-qwen3-4b-ff-reward-filtered",
    revision="06140ea0d09d4cebbbe0334c3cfa0211ff368318",
    local_dir="restore_qwen3_coconut_ff_reward_filtered_ckpt24",
)
```

The exact raw checkpoint will be at:

```text
restore_qwen3_coconut_ff_reward_filtered_ckpt24/latent_checkpoint.pt
```

### `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_26`

- HF repo: `agurung/coconut-qwen3-4b-ff-reward-filtered`
- Exact revision: `b8343ad1e19011881f8d42b5c7e55835fb9fa3a2`
- HF refs observed after publish: `checkpoint_26`, `checkpoint-26`
- Local checkpoint SHA256:
  `54e5b4774d91fb5819c38d542ae0b6b54d54f47303a9f67d33332c55371d9154`
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
    repo_id="agurung/coconut-qwen3-4b-ff-reward-filtered",
    revision="b8343ad1e19011881f8d42b5c7e55835fb9fa3a2",
    local_dir="restore_qwen3_coconut_ff_reward_filtered_ckpt26",
)
```

The exact raw checkpoint will be at:

```text
restore_qwen3_coconut_ff_reward_filtered_ckpt26/latent_checkpoint.pt
```

### `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_32`

- HF repo: `agurung/coconut-qwen3-4b-ff-reward-filtered`
- Exact revision: `72a40dad5aa3bbcc658fb4b706cc7a3ddee50ec2`
- HF refs observed after publish: `checkpoint_32`, `checkpoint-32`
- Local checkpoint SHA256:
  `83b23e020f29c0f5098c5714d162a3e964fdeff76d1a5cbc8f749f67b76cd893`
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
    repo_id="agurung/coconut-qwen3-4b-ff-reward-filtered",
    revision="72a40dad5aa3bbcc658fb4b706cc7a3ddee50ec2",
    local_dir="restore_qwen3_coconut_ff_reward_filtered_ckpt32",
)
```

The exact raw checkpoint will be at:

```text
restore_qwen3_coconut_ff_reward_filtered_ckpt32/latent_checkpoint.pt
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

### `checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_20`

- HF repo: `agurung/coconut-gemma-3-4b-ff-reward-filtered`
- Exact revision: `6d84f5c12b086405382301e2335d4193fa9b7766`
- HF refs observed after publish: `checkpoint_20`, `checkpoint-20`
- Local checkpoint SHA256:
  `8d52bc8389a7f46a84b730a331c4a1fb2d4829a05f24cbbcbb798e988a80dc5b`
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
    repo_id="agurung/coconut-gemma-3-4b-ff-reward-filtered",
    revision="6d84f5c12b086405382301e2335d4193fa9b7766",
    local_dir="restore_gemma3_coconut_ff_reward_filtered_ckpt20",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_coconut_ff_reward_filtered_ckpt20/latent_checkpoint.pt
```

### `checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_24`

- HF repo: `agurung/coconut-gemma-3-4b-ff-reward-filtered`
- Exact revision: `eff3f2d4f3939d37a5a18d1819a843c9633f3799`
- HF refs observed after publish: `checkpoint_24`, `checkpoint-24`
- Local checkpoint SHA256:
  `425d0c3554e266f7a888dc8f353d75185ebf3cfd255d8cae7e5511c28b4335fa`
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
    repo_id="agurung/coconut-gemma-3-4b-ff-reward-filtered",
    revision="eff3f2d4f3939d37a5a18d1819a843c9633f3799",
    local_dir="restore_gemma3_coconut_ff_reward_filtered_ckpt24",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_coconut_ff_reward_filtered_ckpt24/latent_checkpoint.pt
```

### `checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_28`

- HF repo: `agurung/coconut-gemma-3-4b-ff-reward-filtered`
- Exact revision: `057a99af203c4ec7649504cefe0559ea5ad5bba8`
- HF refs observed after publish: `checkpoint_28`, `checkpoint-28`
- Local checkpoint SHA256:
  `cada4436e2305a6b69a85a2ad7ed12ec36aa731ff4a4d141eb1d3d93a474d2e4`
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
    repo_id="agurung/coconut-gemma-3-4b-ff-reward-filtered",
    revision="057a99af203c4ec7649504cefe0559ea5ad5bba8",
    local_dir="restore_gemma3_coconut_ff_reward_filtered_ckpt28",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_coconut_ff_reward_filtered_ckpt28/latent_checkpoint.pt
```

### `checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_32`

- HF repo: `agurung/coconut-gemma-3-4b-ff-reward-filtered`
- Exact revision: `351cb7c2eb791d0713b9062127da234a637feab6`
- HF refs observed after publish: `checkpoint_32`, `checkpoint-32`
- Local checkpoint SHA256:
  `d3dd05f74ed40e7f52ef0dacb5a9d6aa4a1a6f990c7a21ba4a72f856fc923b67`
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
    repo_id="agurung/coconut-gemma-3-4b-ff-reward-filtered",
    revision="351cb7c2eb791d0713b9062127da234a637feab6",
    local_dir="restore_gemma3_coconut_ff_reward_filtered_ckpt32",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_coconut_ff_reward_filtered_ckpt32/latent_checkpoint.pt
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

### `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_8`

- HF repo: `agurung/coconut-gemma-3-1b-gsm-hard`
- Exact revision: `2828c87cd31bdba3288fffc64802d08b520da2ca`
- HF refs observed after publish: `checkpoint_8`, `checkpoint-8`
- Local checkpoint SHA256:
  `12e59cf2f87d0d15d95da3462534143e49b21b230dd5890fc3388aef5a731f1e`
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
    revision="2828c87cd31bdba3288fffc64802d08b520da2ca",
    local_dir="restore_gemma3_1b_coconut_gsm_hard_ckpt8",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_1b_coconut_gsm_hard_ckpt8/latent_checkpoint.pt
```

### `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_12`

- HF repo: `agurung/coconut-gemma-3-1b-gsm-hard`
- Exact revision: `8067349326c3ebf4a07d56a7ba5a33e956487b5e`
- HF refs observed after publish: `checkpoint_12`, `checkpoint-12`
- Local checkpoint SHA256:
  `a557beccce7615e6882c30a6096917fa2f741dafe1c51c1457e918bebf51f600`
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
    revision="8067349326c3ebf4a07d56a7ba5a33e956487b5e",
    local_dir="restore_gemma3_1b_coconut_gsm_hard_ckpt12",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_1b_coconut_gsm_hard_ckpt12/latent_checkpoint.pt
```

### `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_24`

- HF repo: `agurung/coconut-gemma-3-1b-gsm-hard`
- Exact revision: `44dc3da36808649c450c11f34ae4bdc9aab75296`
- HF refs observed after publish: `checkpoint_24`, `checkpoint-24`
- Local checkpoint SHA256:
  `9193402044494d2038b238cbfa0226b5a6b35f194f5f1c01613007274bf2517a`
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
    revision="44dc3da36808649c450c11f34ae4bdc9aab75296",
    local_dir="restore_gemma3_1b_coconut_gsm_hard_ckpt24",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_1b_coconut_gsm_hard_ckpt24/latent_checkpoint.pt
```

### `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_32`

- HF repo: `agurung/coconut-gemma-3-1b-gsm-hard`
- Exact revision: `843988d1b78f17f4734217995b4d2999161bddc5`
- HF refs observed after publish: `checkpoint_32`, `checkpoint-32`
- Local checkpoint SHA256:
  `dd2de65aa8bf89cf61c1cea26c392caabdca900a799c216b13c8d0f0a9328bc8`
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
    revision="843988d1b78f17f4734217995b4d2999161bddc5",
    local_dir="restore_gemma3_1b_coconut_gsm_hard_ckpt32",
)
```

The exact raw checkpoint will be at:

```text
restore_gemma3_1b_coconut_gsm_hard_ckpt32/latent_checkpoint.pt
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

- `checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu`: `checkpoint_2`, `checkpoint_4`, `checkpoint_6`, `checkpoint_10`, `checkpoint_14`, `checkpoint_16`, `checkpoint_18`, `checkpoint_28`, `checkpoint_30`
- `checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard`: `checkpoint_16`, `checkpoint_20`, `checkpoint_28`

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
  checkpoints/qwen3-coconut-ff-v3/checkpoint_24 \
  agurung/coconut-qwen3-4b-ff \
  ab2ee7618ea3d87be00d7f1de9328a9144abb48e
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_20 \
  agurung/coconut-qwen3-4b-ff-reward-filtered \
  7b67e6f88823ece6ed7be74d1847719044f98efc
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_22 \
  agurung/coconut-qwen3-4b-ff-reward-filtered \
  90d513c9b995ddb309a9d5b8735a02c6008b49a8
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_24 \
  agurung/coconut-qwen3-4b-ff-reward-filtered \
  06140ea0d09d4cebbbe0334c3cfa0211ff368318
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_26 \
  agurung/coconut-qwen3-4b-ff-reward-filtered \
  b8343ad1e19011881f8d42b5c7e55835fb9fa3a2
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/qwen3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_32 \
  agurung/coconut-qwen3-4b-ff-reward-filtered \
  72a40dad5aa3bbcc658fb4b706cc7a3ddee50ec2
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_20 \
  agurung/coconut-gemma-3-4b-ff-reward-filtered \
  6d84f5c12b086405382301e2335d4193fa9b7766
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_24 \
  agurung/coconut-gemma-3-4b-ff-reward-filtered \
  eff3f2d4f3939d37a5a18d1819a843c9633f3799
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_28 \
  agurung/coconut-gemma-3-4b-ff-reward-filtered \
  057a99af203c4ec7649504cefe0559ea5ad5bba8
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gemma/gemma3-coconut-ff-reward-filtered-v1-2gpu/checkpoint_32 \
  agurung/coconut-gemma-3-4b-ff-reward-filtered \
  351cb7c2eb791d0713b9062127da234a637feab6
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_4 \
  agurung/coconut-gemma-3-1b-gsm-hard \
  4c2bdcaa2934616d2f0c7a0f1d0f5ba611ec5db5
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_8 \
  agurung/coconut-gemma-3-1b-gsm-hard \
  2828c87cd31bdba3288fffc64802d08b520da2ca
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_12 \
  agurung/coconut-gemma-3-1b-gsm-hard \
  8067349326c3ebf4a07d56a7ba5a33e956487b5e
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_24 \
  agurung/coconut-gemma-3-1b-gsm-hard \
  44dc3da36808649c450c11f34ae4bdc9aab75296
```

and:

```bash
/mnt/disk/litereason_anon/.venv/bin/python \
  scripts/checks/verify_coconut_hf_roundtrip.py \
  checkpoints/gsm_hard/gemma3-1b-coconut-gsm-hard/checkpoint_32 \
  agurung/coconut-gemma-3-1b-gsm-hard \
  843988d1b78f17f4734217995b4d2999161bddc5
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

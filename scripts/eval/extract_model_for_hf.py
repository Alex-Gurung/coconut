#!/usr/bin/env python3
"""
Export a Coconut checkpoint into a faithful Hugging Face package.

The exported directory contains:
- a standard HF model + tokenizer for normal Transformers inference
- tokenizer entries for Coconut latent tokens
- optional `latent_checkpoint.pt` with the original Coconut checkpoint for
  exact wrapper-state recovery and latent decoding
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import torch


LATENT_TOKENS: Tuple[str, str, str] = (
    "<|start-latent|>",
    "<|end-latent|>",
    "<|latent|>",
)
CHECKPOINT_NAME = "latent_checkpoint.pt"
WRAPPER_PREFIXES = ("module.", "_orig_mod.")
EMBED_KEY_CANDIDATES = (
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
)
LM_HEAD_KEY_CANDIDATES = (
    "lm_head.weight",
    "model.language_model.lm_head.weight",
)


def normalize_checkpoint_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Strip DDP / compile wrappers from checkpoint keys."""
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        changed = True
        while changed:
            changed = False
            for prefix in WRAPPER_PREFIXES:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    changed = True
        normalized[normalized_key] = value
    return normalized


def split_coconut_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a Coconut checkpoint into base-model weights and wrapper-only keys."""
    base_model_state: Dict[str, torch.Tensor] = {}
    aux_state: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("base_causallm."):
            base_model_state[key[len("base_causallm.") :]] = value
        else:
            aux_state[key] = value

    return base_model_state, aux_state


def find_first_present_key(
    candidates: Iterable[str],
    mapping: Mapping[str, torch.Tensor],
) -> str | None:
    for candidate in candidates:
        if candidate in mapping:
            return candidate
    return None


def checkpoint_vocab_size(base_model_state: Mapping[str, torch.Tensor]) -> Tuple[str, int]:
    embed_key = find_first_present_key(EMBED_KEY_CANDIDATES, base_model_state)
    if embed_key is None:
        raise KeyError(
            "Could not find an embedding matrix in the extracted base model state."
        )
    return embed_key, int(base_model_state[embed_key].shape[0])


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def add_latent_tokens(tokenizer, expected_vocab_size: int | None = None) -> Dict[str, int]:
    """Mirror training-time tokenizer growth and optionally validate final size."""
    tokenizer.add_tokens(list(LATENT_TOKENS))
    token_ids = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in LATENT_TOKENS}

    for token, token_id in token_ids.items():
        if token_id is None or token_id < 0:
            raise ValueError(f"Tokenizer failed to register latent token {token}.")

    if expected_vocab_size is not None and len(tokenizer) != expected_vocab_size:
        raise ValueError(
            "Tokenizer length does not match checkpoint embedding size after adding "
            f"latent tokens: tokenizer={len(tokenizer)}, checkpoint={expected_vocab_size}"
        )

    return token_ids


def link_or_copy(src: Path, dst: Path) -> str:
    """Stage a large checkpoint efficiently, preferring hard links over copies."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build_readme(
    model_id: str,
    checkpoint_name: str,
    include_latent_checkpoint: bool,
    latent_token_ids: Mapping[str, int],
    aux_state_keys: Iterable[str],
) -> str:
    aux_lines = "\n".join(f"- `{key}`" for key in sorted(aux_state_keys))
    latent_section = (
        "### Exact Coconut latent decoding\n\n"
        "The export also includes `latent_checkpoint.pt`, which is the original Coconut "
        "checkpoint. That file preserves the wrapper state needed for exact latent "
        "decoding and checkpoint-faithful recovery.\n"
    )
    if not include_latent_checkpoint:
        latent_section = (
            "### Exact Coconut latent decoding\n\n"
            "This export does not include `latent_checkpoint.pt`, so it is limited to "
            "standard HF loading and does not provide exact raw-checkpoint recovery.\n"
        )

    latent_id_lines = "\n".join(
        f"- `{token}` -> `{token_id}`" for token, token_id in latent_token_ids.items()
    )

    return f"""# Coconut Export For {model_id}

This package was exported from Coconut checkpoint `{checkpoint_name}`.

## What Is Included

- Standard HF model weights for `AutoModelForCausalLM.from_pretrained(...)`
- A tokenizer that includes the Coconut latent tokens used during training
- Metadata describing the latent-token ids and auxiliary Coconut state

{latent_section}

## Latent Tokens

{latent_id_lines}

## Coconut-Only Auxiliary State

The original Coconut checkpoint contains wrapper-only keys in addition to the base LM:

{aux_lines}

## Standard Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("PATH_OR_HF_REPO")
tokenizer = AutoTokenizer.from_pretrained("PATH_OR_HF_REPO")
```
"""


def write_metadata(
    output_dir: Path,
    checkpoint_path: Path,
    model_id: str,
    latent_token_ids: Mapping[str, int],
    aux_state: Mapping[str, torch.Tensor],
    include_latent_checkpoint: bool,
    latent_checkpoint_mode: str | None,
) -> Path:
    metadata = {
        "method": "coconut",
        "base_model": model_id,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "latent_tokens": list(LATENT_TOKENS),
        "latent_token_ids": dict(latent_token_ids),
        "raw_checkpoint_file": CHECKPOINT_NAME if include_latent_checkpoint else None,
        "raw_checkpoint_mode": latent_checkpoint_mode,
        "aux_state_keys": sorted(aux_state.keys()),
        "export_script": "scripts/eval/extract_model_for_hf.py",
    }
    metadata_path = output_dir / "latent_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path


def rewrite_saved_config_version(output_dir: Path, source_transformers_version: str | None) -> None:
    if not source_transformers_version:
        return
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    config["transformers_version"] = source_transformers_version
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def rewrite_saved_tokenizer_config(output_dir: Path, tokenizer) -> None:
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return

    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template and "chat_template" not in tokenizer_config:
        tokenizer_config["chat_template"] = chat_template
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")


def export_coconut_model(
    checkpoint_path: str,
    output_dir: str,
    model_id: str,
    include_latent_checkpoint: bool = True,
    safe_serialization: bool = True,
) -> Path:
    """
    Export a Coconut checkpoint into a faithful HF package.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path_obj = Path(checkpoint_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path_obj}")
    checkpoint = torch.load(checkpoint_path_obj, map_location="cpu")
    stripped = normalize_checkpoint_state_dict(checkpoint)
    base_model_state, aux_state = split_coconut_state_dict(stripped)

    if not base_model_state:
        raise ValueError("Checkpoint does not contain any `base_causallm.` weights.")

    embed_key, expected_vocab_size = checkpoint_vocab_size(base_model_state)
    lm_head_key = find_first_present_key(LM_HEAD_KEY_CANDIDATES, base_model_state)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ensure_pad_token(tokenizer)
    latent_token_ids = add_latent_tokens(tokenizer, expected_vocab_size=expected_vocab_size)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    source_transformers_version = getattr(base_model.config, "transformers_version", None)

    model_state = base_model.state_dict()
    model_embed_key = find_first_present_key(EMBED_KEY_CANDIDATES, model_state)
    if model_embed_key is None:
        raise KeyError("Loaded base model does not expose a known embedding matrix.")

    model_vocab_size = int(model_state[model_embed_key].shape[0])
    if model_vocab_size != expected_vocab_size:
        print(
            "Resizing token embeddings from "
            f"{model_vocab_size} to checkpoint size {expected_vocab_size}."
        )
        base_model.resize_token_embeddings(expected_vocab_size, mean_resizing=False)

    load_result = base_model.load_state_dict(base_model_state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint did not load cleanly into the base model.\n"
            f"Missing keys: {load_result.missing_keys}\n"
            f"Unexpected keys: {load_result.unexpected_keys}"
        )

    print(f"Saving model to: {output_dir_obj}")
    base_model.save_pretrained(output_dir_obj, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir_obj)
    rewrite_saved_config_version(output_dir_obj, source_transformers_version)
    rewrite_saved_tokenizer_config(output_dir_obj, tokenizer)

    latent_checkpoint_mode = None
    if include_latent_checkpoint:
        latent_checkpoint_path = output_dir_obj / CHECKPOINT_NAME
        latent_checkpoint_mode = link_or_copy(checkpoint_path_obj, latent_checkpoint_path)
        print(
            f"Staged raw Coconut checkpoint at {latent_checkpoint_path} "
            f"using {latent_checkpoint_mode}."
        )

    write_metadata(
        output_dir=output_dir_obj,
        checkpoint_path=checkpoint_path_obj,
        model_id=model_id,
        latent_token_ids=latent_token_ids,
        aux_state=aux_state,
        include_latent_checkpoint=include_latent_checkpoint,
        latent_checkpoint_mode=latent_checkpoint_mode,
    )

    readme_path = output_dir_obj / "README.md"
    readme_path.write_text(
        build_readme(
            model_id=model_id,
            checkpoint_name=checkpoint_path_obj.name,
            include_latent_checkpoint=include_latent_checkpoint,
            latent_token_ids=latent_token_ids,
            aux_state_keys=aux_state.keys(),
        )
    )

    print(
        "✓ Export complete.\n"
        f"  Embedding key: {embed_key}\n"
        f"  Vocab size: {expected_vocab_size}\n"
        f"  LM head key: {lm_head_key}\n"
        f"  Latent token ids: {latent_token_ids}"
    )
    return output_dir_obj


def extract_coconut_model(
    checkpoint_path: str,
    output_dir: str,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    include_latent_checkpoint: bool = True,
    safe_serialization: bool = True,
) -> Path:
    """Backward-compatible wrapper for older callers."""
    return export_coconut_model(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        model_id=model_id,
        include_latent_checkpoint=include_latent_checkpoint,
        safe_serialization=safe_serialization,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", help="Path to Coconut checkpoint")
    parser.add_argument("output_dir", help="Output directory for HF model package")
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model ID used to reconstruct the architecture and tokenizer",
    )
    parser.add_argument(
        "--skip-latent-checkpoint",
        action="store_true",
        help="Do not bundle the raw Coconut checkpoint as latent_checkpoint.pt",
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Save PyTorch binaries instead of safetensors",
    )
    args = parser.parse_args()

    export_coconut_model(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        include_latent_checkpoint=not args.skip_latent_checkpoint,
        safe_serialization=not args.no_safe_serialization,
    )


if __name__ == "__main__":
    main()

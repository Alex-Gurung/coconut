#!/usr/bin/env python3
"""
Verify that an HF snapshot faithfully reproduces a local Coconut checkpoint.

Checks performed:
- downloaded `latent_checkpoint.pt` matches the local checkpoint by SHA256
- every base-model tensor in the HF export matches the local checkpoint
- the tokenizer contains the Coconut latent tokens
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.extract_model_for_hf import (
    LATENT_TOKENS,
    CHECKPOINT_NAME,
    normalize_checkpoint_state_dict,
    split_coconut_state_dict,
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_model_state(local_checkpoint: Path, downloaded_dir: Path) -> dict:
    checkpoint = torch.load(local_checkpoint, map_location="cpu")
    base_state, aux_state = split_coconut_state_dict(
        normalize_checkpoint_state_dict(checkpoint)
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(downloaded_dir, torch_dtype="auto")
    except ValueError as exc:
        config = json.loads((downloaded_dir / "config.json").read_text())
        model_type = config.get("model_type", "<unknown>")
        raise RuntimeError(
            "Could not load the exported HF model for round-trip verification. "
            f"Transformers {transformers_version} does not recognize model_type={model_type!r}. "
            "Re-run the verifier in an environment with a newer Transformers build that "
            "supports this architecture."
        ) from exc
    model_state = model.state_dict()

    missing = sorted(set(base_state) - set(model_state))
    unexpected = sorted(set(model_state) - set(base_state))
    diff_keys = []

    for key in sorted(base_state):
        if key not in model_state:
            diff_keys.append(key)
            continue
        src = base_state[key]
        dst = model_state[key]
        if src.shape != dst.shape or src.dtype != dst.dtype or not torch.equal(src, dst):
            diff_keys.append(key)
            if len(diff_keys) >= 20:
                break

    return {
        "base_key_count": len(base_state),
        "aux_keys": sorted(aux_state.keys()),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "diff_count": len(diff_keys),
        "diff_examples": diff_keys[:20],
        "all_equal": not missing and not unexpected and not diff_keys,
    }


def inspect_tokenizer(downloaded_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(downloaded_dir)
    token_ids = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in LATENT_TOKENS}
    return {
        "vocab_size": len(tokenizer),
        "latent_token_ids": token_ids,
        "all_present": all(token_id >= 0 for token_id in token_ids.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_checkpoint", help="Path to the local Coconut checkpoint")
    parser.add_argument("repo_id", help="HF repo id")
    parser.add_argument("revision", help="Exact HF revision/commit SHA to verify")
    parser.add_argument(
        "--download-dir",
        help="Optional explicit download directory",
    )
    args = parser.parse_args()

    local_checkpoint = Path(args.local_checkpoint).resolve()
    download_dir = Path(args.download_dir).resolve() if args.download_dir else None
    if download_dir and download_dir.exists():
        shutil.rmtree(download_dir)

    downloaded = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(download_dir) if download_dir else None,
        )
    ).resolve()

    latent_checkpoint = downloaded / CHECKPOINT_NAME
    result = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "downloaded_dir": str(downloaded),
        "local_checkpoint": str(local_checkpoint),
        "local_checkpoint_sha256": sha256sum(local_checkpoint),
        "downloaded_latent_checkpoint_present": latent_checkpoint.exists(),
        "downloaded_latent_checkpoint_sha256": sha256sum(latent_checkpoint)
        if latent_checkpoint.exists()
        else None,
        "latent_checkpoint_matches": latent_checkpoint.exists()
        and sha256sum(local_checkpoint) == sha256sum(latent_checkpoint),
        "metadata_present": (downloaded / "latent_metadata.json").exists(),
        "readme_present": (downloaded / "README.md").exists(),
    }
    result["model"] = compare_model_state(local_checkpoint, downloaded)
    result["tokenizer"] = inspect_tokenizer(downloaded)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Inventory local Coconut checkpoints and discover exact HF recovery paths.

For each local `checkpoints/**/checkpoint_*` file, this script reports whether:
- the file is still a real local checkpoint or an HF offload placeholder
- any public HF repo under the target author exposes `latent_metadata.json`
  pointing back to that exact local checkpoint path
- any matched HF export can be round-trip verified with the repo-local verifier
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from huggingface_hub import HfApi, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.extract_model_for_hf import CHECKPOINT_NAME


def iter_local_checkpoints(root: Path) -> List[Path]:
    return sorted(path.resolve() for path in root.glob("**/checkpoint_*") if path.is_file())


def is_placeholder_checkpoint(path: Path) -> bool:
    if path.stat().st_size > 4096:
        return False
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return False
    return "HF offload placeholder" in text


def unique_ref_names(branches, tags) -> List[str]:
    ref_names: List[str] = []
    seen = set()
    for name in ["main"] + [branch.name for branch in branches] + [tag.name for tag in tags]:
        if name not in seen:
            ref_names.append(name)
            seen.add(name)
    return ref_names


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def gather_hf_checkpoint_hits(author: str, limit: int) -> Dict[Path, List[dict]]:
    api = HfApi()
    hits_by_checkpoint: Dict[Path, List[dict]] = defaultdict(list)

    for model in api.list_models(author=author, full=False, limit=limit):
        repo_id = model.id
        try:
            refs = api.list_repo_refs(repo_id, repo_type="model")
        except Exception:
            continue

        branches = getattr(refs, "branches", [])
        tags = getattr(refs, "tags", [])
        ref_to_sha = {ref.name: ref.target_commit for ref in [*branches, *tags]}

        for ref_name in unique_ref_names(branches, tags):
            try:
                metadata_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="latent_metadata.json",
                    revision=ref_name,
                )
            except Exception:
                continue

            try:
                metadata = json.loads(Path(metadata_path).read_text())
            except json.JSONDecodeError:
                continue

            checkpoint_path = metadata.get("checkpoint_path")
            if not checkpoint_path:
                continue

            info = api.model_info(repo_id, revision=ref_name)
            hit = {
                "repo_id": repo_id,
                "ref": ref_name,
                "exact_sha": ref_to_sha.get(ref_name, info.sha),
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "checkpoint_name": metadata.get("checkpoint_name"),
                "raw_checkpoint_file": metadata.get("raw_checkpoint_file"),
                "latent_checkpoint_present": any(
                    sibling.rfilename == CHECKPOINT_NAME for sibling in info.siblings
                ),
            }
            hits_by_checkpoint[Path(checkpoint_path).resolve()].append(hit)

    deduped_hits: Dict[Path, List[dict]] = {}
    for checkpoint_path, hits in hits_by_checkpoint.items():
        deduped: Dict[tuple[str, str], dict] = {}
        for hit in hits:
            key = (hit["repo_id"], hit["exact_sha"])
            existing = deduped.get(key)
            if existing is None:
                hit["refs"] = [hit["ref"]]
                hit.pop("ref", None)
                deduped[key] = hit
                continue
            existing["refs"].append(hit["ref"])
            existing["latent_checkpoint_present"] = (
                existing["latent_checkpoint_present"] or hit["latent_checkpoint_present"]
            )
            if existing.get("raw_checkpoint_file") is None:
                existing["raw_checkpoint_file"] = hit.get("raw_checkpoint_file")
        for hit in deduped.values():
            hit["refs"] = sorted(set(hit["refs"]))
        deduped_hits[checkpoint_path] = sorted(
            deduped.values(), key=lambda item: (item["repo_id"], item["exact_sha"])
        )
    return deduped_hits


def verify_hf_match(local_checkpoint: Path, match: dict, download_root: Path) -> dict:
    from huggingface_hub import snapshot_download

    from scripts.checks.verify_coconut_hf_roundtrip import (
        compare_model_state,
        inspect_tokenizer,
        sha256sum,
    )

    target_dir = download_root / (
        f"{slugify(match['repo_id'])}_{slugify(match['exact_sha'])}_{local_checkpoint.name}"
    )
    if target_dir.exists():
        shutil.rmtree(target_dir)

    downloaded_dir = Path(
        snapshot_download(
            repo_id=match["repo_id"],
            revision=match["exact_sha"],
            local_dir=str(target_dir),
        )
    ).resolve()

    latent_checkpoint = downloaded_dir / CHECKPOINT_NAME
    result = {
        "repo_id": match["repo_id"],
        "exact_sha": match["exact_sha"],
        "downloaded_dir": str(downloaded_dir),
        "downloaded_latent_checkpoint_present": latent_checkpoint.exists(),
        "downloaded_latent_checkpoint_sha256": sha256sum(latent_checkpoint)
        if latent_checkpoint.exists()
        else None,
        "local_checkpoint_sha256": sha256sum(local_checkpoint),
    }
    result["latent_checkpoint_matches"] = (
        latent_checkpoint.exists()
        and result["downloaded_latent_checkpoint_sha256"] == result["local_checkpoint_sha256"]
    )
    result["metadata_present"] = (downloaded_dir / "latent_metadata.json").exists()
    result["readme_present"] = (downloaded_dir / "README.md").exists()
    result["model"] = compare_model_state(local_checkpoint, downloaded_dir)
    result["tokenizer"] = inspect_tokenizer(downloaded_dir)
    result["ok"] = (
        result["latent_checkpoint_matches"]
        and result["model"]["all_equal"]
        and result["tokenizer"]["all_present"]
        and result["metadata_present"]
        and result["readme_present"]
    )
    return result


def audit_local_checkpoints(
    checkpoints_root: Path,
    author: str,
    limit: int,
    verify_matches: bool,
    download_root: Path | None,
) -> dict:
    local_checkpoints = iter_local_checkpoints(checkpoints_root)
    hf_hits = gather_hf_checkpoint_hits(author=author, limit=limit)

    results = []
    for checkpoint_path in local_checkpoints:
        placeholder = is_placeholder_checkpoint(checkpoint_path)
        matches = hf_hits.get(checkpoint_path, [])
        record = {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "placeholder": placeholder,
            "hf_matches": matches,
        }

        if placeholder and matches:
            record["status"] = "hf_placeholder_offloaded"
        elif matches:
            record["status"] = "hf_candidate_found"
        else:
            record["status"] = "no_hf_recovery_path_found"

        if verify_matches and matches and not placeholder:
            if download_root is None:
                raise ValueError("--download-root is required with --verify-matches")
            verification_results = [
                verify_hf_match(checkpoint_path, match, download_root) for match in matches
            ]
            record["verification"] = verification_results
            if verification_results and all(result["ok"] for result in verification_results):
                record["status"] = "exact_roundtrip_verified"
            else:
                record["status"] = "hf_candidate_verification_failed"

        results.append(record)

    summary = {
        "author": author,
        "checkpoint_count": len(results),
        "status_counts": dict(sorted(_count_statuses(results).items())),
        "results": results,
    }
    return summary


def _count_statuses(results: Iterable[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for result in results:
        counts[result["status"]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints-root",
        default="checkpoints",
        help="Root directory containing local checkpoints",
    )
    parser.add_argument(
        "--author",
        default="agurung",
        help="HF author/org to scan for latent_metadata-backed exports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of HF repos to inspect for the target author",
    )
    parser.add_argument(
        "--verify-matches",
        action="store_true",
        help="Run full round-trip verification for matched non-placeholder checkpoints",
    )
    parser.add_argument(
        "--download-root",
        help="Scratch directory for downloaded HF snapshots when verifying matches",
    )
    args = parser.parse_args()

    download_root = Path(args.download_root).resolve() if args.download_root else None
    if download_root:
        download_root.mkdir(parents=True, exist_ok=True)

    summary = audit_local_checkpoints(
        checkpoints_root=Path(args.checkpoints_root).resolve(),
        author=args.author,
        limit=args.limit,
        verify_matches=args.verify_matches,
        download_root=download_root,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

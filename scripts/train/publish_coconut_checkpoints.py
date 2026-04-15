#!/usr/bin/env python3
"""
Export, upload, and verify Coconut checkpoints listed in the publish manifest.

The intended workflow is:
1. export a faithful HF package with tokenizer growth and latent checkpoint bundle
2. upload it to a checkpoint-specific HF branch and tag
3. verify the uploaded immutable commit SHA against the local checkpoint
4. keep only the balanced local subset after the archive is complete
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.checks.verify_coconut_hf_roundtrip import (  # noqa: E402
    CHECKPOINT_NAME,
    compare_model_state,
    inspect_tokenizer,
    sha256sum,
)
from scripts.eval.extract_model_for_hf import export_coconut_model  # noqa: E402
from scripts.train.upload_to_hf import upload_model_to_hf  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "scripts/checks/checkpoint_publish_manifest.json"
DEFAULT_STAGING_ROOT = Path("/tmp/coconut_hf_publish")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def select_entries(manifest: dict, profile: str, contains: str | None) -> list[dict]:
    entries = [
        entry
        for entry in manifest["entries"]
        if profile in entry.get("profiles", [manifest.get("default_publish_profile")])
    ]
    if contains:
        entries = [
            entry
            for entry in entries
            if contains in entry["checkpoint"]
            or contains in entry["repo_id"]
            or contains in entry["revision"]
        ]
    return entries


def sanitize_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def verify_uploaded_checkpoint(
    local_checkpoint: Path,
    repo_id: str,
    revision: str,
    download_dir: Path,
) -> dict[str, Any]:
    if download_dir.exists():
        shutil.rmtree(download_dir)

    downloaded = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(download_dir),
        )
    ).resolve()

    latent_checkpoint = downloaded / CHECKPOINT_NAME
    result = {
        "repo_id": repo_id,
        "revision": revision,
        "downloaded_dir": str(downloaded),
        "local_checkpoint": str(local_checkpoint),
        "local_checkpoint_sha256": sha256sum(local_checkpoint),
        "downloaded_latent_checkpoint_present": latent_checkpoint.exists(),
        "downloaded_latent_checkpoint_sha256": sha256sum(latent_checkpoint)
        if latent_checkpoint.exists()
        else None,
        "latent_checkpoint_matches": latent_checkpoint.exists()
        and sha256sum(local_checkpoint) == sha256sum(latent_checkpoint),
    }
    result["model"] = compare_model_state(local_checkpoint, downloaded)
    result["tokenizer"] = inspect_tokenizer(downloaded)
    result["all_equal"] = (
        result["latent_checkpoint_matches"]
        and result["model"]["all_equal"]
        and result["tokenizer"]["all_present"]
    )
    return result


def create_refs_only(entry: dict, private: bool) -> dict[str, Any]:
    repo_id = entry["repo_id"]
    source_revision = entry["source_revision"]
    branch = entry["revision"]
    tag = entry.get("tag")

    api = HfApi()
    create_repo(repo_id, private=private, exist_ok=True)
    api.create_branch(repo_id, repo_type="model", branch=branch, revision=source_revision, exist_ok=True)
    if tag:
        try:
            api.delete_tag(repo_id, repo_type="model", tag=tag)
        except Exception:
            pass
        api.create_tag(
            repo_id,
            repo_type="model",
            tag=tag,
            revision=source_revision,
            exist_ok=True,
        )
    return {
        "action": "create_refs_only",
        "repo_id": repo_id,
        "revision": branch,
        "tag": tag,
        "source_revision": source_revision,
    }


def export_upload_verify(
    entry: dict,
    staging_root: Path,
    keep_staging: bool,
    private: bool,
) -> dict[str, Any]:
    checkpoint = (REPO_ROOT / entry["checkpoint"]).resolve()
    repo_id = entry["repo_id"]
    revision = entry["revision"]
    tag = entry.get("tag")
    username, repo_name = repo_id.split("/", 1)

    export_dir = staging_root / f"{sanitize_name(repo_id)}__{revision}"
    verify_dir = staging_root / f"{sanitize_name(repo_id)}__{revision}__verify"

    if export_dir.exists():
        shutil.rmtree(export_dir)

    export_coconut_model(
        checkpoint_path=str(checkpoint),
        output_dir=str(export_dir),
        model_id=entry["model_id"],
        include_latent_checkpoint=True,
        safe_serialization=True,
    )

    upload_result = upload_model_to_hf(
        model_dir=str(export_dir),
        repo_name=repo_name,
        username=username,
        private=private,
        revision=revision,
        tag=tag,
        commit_message=f"Add faithful Coconut export for {checkpoint.name}",
    )
    if upload_result is None:
        raise RuntimeError(f"Upload failed for {checkpoint}")

    verify_result = verify_uploaded_checkpoint(
        local_checkpoint=checkpoint,
        repo_id=repo_id,
        revision=upload_result["commit_oid"],
        download_dir=verify_dir,
    )
    if not verify_result["all_equal"]:
        raise RuntimeError(
            "Round-trip verification failed after upload.\n"
            f"{json.dumps(verify_result, indent=2)}"
        )

    if not keep_staging:
        shutil.rmtree(export_dir, ignore_errors=True)
        shutil.rmtree(verify_dir, ignore_errors=True)

    return {
        "action": "export_upload_verify",
        "checkpoint": entry["checkpoint"],
        "repo_id": repo_id,
        "revision": revision,
        "tag": tag,
        "upload": upload_result,
        "verify": verify_result,
    }


def process_entry(
    entry: dict,
    staging_root: Path,
    keep_staging: bool,
    private: bool,
    execute: bool,
) -> dict[str, Any]:
    result = {
        "checkpoint": entry["checkpoint"],
        "repo_id": entry["repo_id"],
        "revision": entry["revision"],
        "tag": entry.get("tag"),
        "action": entry["action"],
    }
    if not execute:
        result["dry_run"] = True
        return result

    if entry["action"] == "create_refs_only":
        result.update(create_refs_only(entry, private=private))
        return result
    if entry["action"] == "export_upload_verify":
        result.update(
            export_upload_verify(
                entry,
                staging_root=staging_root,
                keep_staging=keep_staging,
                private=private,
            )
        )
        return result

    raise ValueError(f"Unknown action: {entry['action']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the publish manifest JSON file",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Publish profile to execute; defaults to manifest.default_publish_profile",
    )
    parser.add_argument(
        "--contains",
        help="Only process entries whose checkpoint/repo/revision contains this substring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of entries to process",
    )
    parser.add_argument(
        "--staging-root",
        default=str(DEFAULT_STAGING_ROOT),
        help="Temporary working directory for exports and verification downloads",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep exported and downloaded staging directories after success",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create destination repos as private if they do not already exist",
    )
    parser.add_argument(
        "--results-json",
        help="Optional path to write the execution results as JSON",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform uploads and ref creation. Without this flag the script is dry-run only.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    profile = args.profile or manifest.get("default_publish_profile", "conservative_archive")
    known_profiles = set(manifest.get("profiles", {}))
    if known_profiles and profile not in known_profiles:
        raise SystemExit(
            f"Unknown profile {profile!r}. Available profiles: {', '.join(sorted(known_profiles))}"
        )

    entries = select_entries(manifest, profile=profile, contains=args.contains)
    if args.limit is not None:
        entries = entries[: args.limit]

    staging_root = Path(args.staging_root).resolve()
    staging_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for entry in entries:
        print(f"Processing {entry['checkpoint']} -> {entry['repo_id']}@{entry['revision']}")
        result = process_entry(
            entry,
            staging_root=staging_root,
            keep_staging=args.keep_staging,
            private=args.private,
            execute=args.execute,
        )
        results.append(result)

    payload = {
        "manifest": str(manifest_path),
        "profile": profile,
        "execute": args.execute,
        "results": results,
    }

    if args.results_json:
        Path(args.results_json).write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

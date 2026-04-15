#!/usr/bin/env python3
"""
Upload an extracted Coconut model package to Hugging Face Hub.

This helper is intended for checkpoint-faithful uploads where we want to:
- push to a dedicated branch such as `checkpoint_24`
- optionally create a stable tag such as `checkpoint-24`
- record the immutable uploaded commit SHA for later round-trip verification
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_model_to_hf(
    model_dir: str,
    repo_name: str,
    username: str = "agurung",
    private: bool = False,
    revision: str | None = None,
    tag: str | None = None,
    commit_message: str | None = None,
    replace_tag: bool = True,
) -> dict | None:
    """
    Upload a prepared model directory to Hugging Face Hub.
    """
    model_dir_path = Path(model_dir).resolve()
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir_path}")

    repo_id = f"{username}/{repo_name}"

    if revision:
        print(f"Uploading model to: https://huggingface.co/{repo_id}/tree/{revision}")
        print(f"Revision: {revision}")
    else:
        print(f"Uploading model to: https://huggingface.co/{repo_id}")
    print(f"Model directory: {model_dir_path}")

    api = HfApi()

    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception as exc:
        raise RuntimeError(
            "Not logged in to Hugging Face Hub. Run `huggingface-cli login` first."
        ) from exc

    create_repo(repo_id, private=private, exist_ok=True)
    print(f"Repository ready: {repo_id}")

    if revision:
        api.create_branch(repo_id, branch=revision, exist_ok=True)
        print(f"Branch ready: {revision}")

    commit = api.upload_folder(
        folder_path=str(model_dir_path),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message or f"Upload {model_dir_path.name}",
    )

    target_revision = revision or commit.oid
    print("Upload completed.")
    print(f"Commit SHA: {commit.oid}")
    print(f"Commit URL: {commit.commit_url}")

    if tag:
        if replace_tag:
            try:
                api.delete_tag(repo_id, repo_type="model", tag=tag)
                print(f"Removed existing tag: {tag}")
            except Exception:
                pass
        api.create_tag(
            repo_id,
            repo_type="model",
            tag=tag,
            revision=commit.oid,
            exist_ok=True,
        )
        print(f"Tag ready: {tag}")

    if revision:
        print(f"Branch URL: https://huggingface.co/{repo_id}/tree/{revision}")
    if tag:
        print(f"Tag URL: https://huggingface.co/{repo_id}/tree/{tag}")
    print(f"Exact revision for verification: {commit.oid}")

    return {
        "repo_id": repo_id,
        "revision": target_revision,
        "commit_oid": commit.oid,
        "commit_url": commit.commit_url,
        "tag": tag,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Directory containing the extracted model")
    parser.add_argument("--repo_name", default="coconut-qwen2.5-7b", help="HF repo name")
    parser.add_argument("--username", default="agurung", help="HF username")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--revision", help="Branch/revision name such as checkpoint_24")
    parser.add_argument("--tag", help="Optional stable tag such as checkpoint-24")
    parser.add_argument("--commit-message", help="Optional upload commit message")
    args = parser.parse_args()

    result = upload_model_to_hf(
        model_dir=args.model_dir,
        repo_name=args.repo_name,
        username=args.username,
        private=args.private,
        revision=args.revision,
        tag=args.tag,
        commit_message=args.commit_message,
        replace_tag=True,
    )
    if result is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

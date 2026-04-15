#!/usr/bin/env python3
"""
Report current HF namespace coverage for Coconut checkpoint upload targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


RELEVANT_SUBSTRINGS = (
    "coconut",
    "flawed-fictions",
    "gsm-hard",
    "colar",
)

SUGGESTED_TARGETS = [
    {
        "label": "Qwen3 Coconut FF standard",
        "repo_id": "agurung/coconut-qwen3-4b-ff",
    },
    {
        "label": "Gemma 3 4B Coconut FF standard",
        "repo_id": "agurung/coconut-gemma-3-4b-ff",
    },
    {
        "label": "Qwen3 Coconut FF reward-filtered",
        "repo_id": "agurung/coconut-qwen3-4b-ff-reward-filtered",
        "related_existing_repos": [
            "agurung/coconut-qwen3-4b-ff",
            "agurung/flawed-fictions-qwen3-4b",
            "agurung/flawed-fictions-qwen3-4b-litereason",
        ],
    },
    {
        "label": "Gemma 3 4B Coconut FF reward-filtered",
        "repo_id": "agurung/coconut-gemma-3-4b-ff-reward-filtered",
        "related_existing_repos": [
            "agurung/coconut-gemma-3-4b-ff",
            "agurung/flawed-fictions-gemma-3-4b",
        ],
    },
    {
        "label": "Gemma 3 1B Coconut GSM-Hard",
        "repo_id": "agurung/coconut-gemma-3-1b-gsm-hard",
        "related_existing_repos": [
            "agurung/colar-gemma-3-1b-gsm-hard-sft",
            "agurung/colar-gemma-3-1b-gsm-hard-rl",
        ],
    },
]


def list_namespace_models(api: HfApi, author: str, limit: int) -> list[str]:
    return sorted(model.id for model in api.list_models(author=author, full=False, limit=limit))


def fetch_refs(api: HfApi, repo_id: str) -> dict:
    refs = api.list_repo_refs(repo_id, repo_type="model")
    return {
        "branches": sorted(branch.name for branch in refs.branches),
        "tags": sorted(tag.name for tag in refs.tags),
    }


def build_report(author: str, limit: int) -> dict:
    api = HfApi()
    namespace_models = list_namespace_models(api, author=author, limit=limit)
    namespace_set = set(namespace_models)
    relevant_models = [
        repo_id
        for repo_id in namespace_models
        if any(token in repo_id.lower() for token in RELEVANT_SUBSTRINGS)
    ]

    targets = []
    for target in SUGGESTED_TARGETS:
        repo_id = target["repo_id"]
        entry = {
            **target,
            "exists": repo_id in namespace_set,
        }
        repos_to_describe = [repo_id, *target.get("related_existing_repos", [])]
        related = []
        for related_repo in repos_to_describe:
            exists = related_repo in namespace_set
            item = {
                "repo_id": related_repo,
                "exists": exists,
            }
            if exists:
                item.update(fetch_refs(api, related_repo))
            related.append(item)
        entry["repo_group"] = related
        targets.append(entry)

    return {
        "author": author,
        "relevant_models": relevant_models,
        "suggested_targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default="agurung")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    print(json.dumps(build_report(author=args.author, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()

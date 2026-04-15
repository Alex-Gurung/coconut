#!/usr/bin/env python3
"""
Dry-run checkpoint pruning planner based on a curated retention manifest.

This script never deletes anything. It only reports:
- which checkpoints are retained for a selected profile
- which currently present checkpoints fall outside the keep set
- the approximate space reclaimed if those prune candidates were removed later
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "scripts/checks/checkpoint_retention_manifest.json"


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_run_checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    return sorted(path for path in run_dir.iterdir() if path.is_file() and path.name.startswith("checkpoint_"))


def build_run_plan(run: dict, profile: str) -> dict:
    run_dir = REPO_ROOT / run["path"]
    present_paths = iter_run_checkpoints(run_dir)
    present = {path.name: path for path in present_paths}
    keep_items = [
        item for item in run["keep"] if profile in item.get("profiles", ["conservative_archive"])
    ]
    keep_names = [item["name"] for item in keep_items]
    keep_set = set(keep_names)

    keep_present = [name for name in keep_names if name in present]
    keep_missing = [name for name in keep_names if name not in present]
    prune_candidates = [name for name in sorted(present) if name not in keep_set]

    size_by_name = {name: present[name].stat().st_size for name in present}
    total_size = sum(size_by_name.values())
    keep_size = sum(size_by_name[name] for name in keep_present)
    prune_size = sum(size_by_name[name] for name in prune_candidates)

    return {
        "path": run["path"],
        "label": run["label"],
        "kind": run["kind"],
        "run_exists": run_dir.exists(),
        "present_checkpoints": sorted(present),
        "keep_present": keep_present,
        "keep_missing": keep_missing,
        "prune_candidates": prune_candidates,
        "total_size_bytes": total_size,
        "keep_size_bytes": keep_size,
        "prune_size_bytes": prune_size,
        "hf_target": run.get("hf_target", {}),
        "notes": run.get("notes", []),
        "alternates": run.get("alternates", []),
        "keep": keep_items,
    }


def print_text_report(manifest: dict, plans: list[dict], profile: str) -> None:
    print("Checkpoint prune plan (dry run only)")
    print(f"Manifest: {DEFAULT_MANIFEST}")
    print(f"Profile: {profile}")
    profile_info = manifest.get("profiles", {}).get(profile, {})
    if profile_info.get("description"):
        print(f"Description: {profile_info['description']}")
    print()

    total_bytes = 0
    keep_bytes = 0
    prune_bytes = 0

    for plan in plans:
        total_bytes += plan["total_size_bytes"]
        keep_bytes += plan["keep_size_bytes"]
        prune_bytes += plan["prune_size_bytes"]

        print(f"{plan['label']}")
        print(f"  path: {plan['path']}")
        print(f"  present: {', '.join(plan['present_checkpoints']) or '(none)'}")
        print(f"  keep: {', '.join(plan['keep_present']) or '(none)'}")
        if plan["keep_missing"]:
            print(f"  keep_missing: {', '.join(plan['keep_missing'])}")
        print(f"  prune_candidates: {', '.join(plan['prune_candidates']) or '(none)'}")
        print(
            "  sizes:"
            f" total={format_bytes(plan['total_size_bytes'])}"
            f" keep={format_bytes(plan['keep_size_bytes'])}"
            f" reclaimable={format_bytes(plan['prune_size_bytes'])}"
        )
        for item in plan["keep"]:
            if item["name"] in plan["keep_present"]:
                print(f"  reason {item['name']}: {item['reason']}")
        for note in plan["notes"]:
            print(f"  note: {note}")
        for alt in plan["alternates"]:
            print(f"  alternate {alt['name']}: {alt['reason']}")
        suggested_repo = plan["hf_target"].get("suggested_repo")
        if suggested_repo:
            print(f"  hf_suggested_repo: {suggested_repo}")
        print()

    print("Totals")
    print(f"  current_size: {format_bytes(total_bytes)}")
    print(f"  retained_size: {format_bytes(keep_bytes)}")
    print(f"  reclaimable_if_pruned: {format_bytes(prune_bytes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the retention manifest JSON file",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Retention profile to evaluate; defaults to manifest.default_local_profile",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text report",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    profile = args.profile or manifest.get("default_local_profile", "balanced_local")
    known_profiles = set(manifest.get("profiles", {}))
    if known_profiles and profile not in known_profiles:
        raise SystemExit(
            f"Unknown profile {profile!r}. Available profiles: {', '.join(sorted(known_profiles))}"
        )
    plans = [build_run_plan(run, profile=profile) for run in manifest["runs"]]

    if args.json:
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "generated_on": manifest.get("generated_on"),
                    "profile": profile,
                    "selection_basis": manifest.get("selection_basis", []),
                    "plans": plans,
                },
                indent=2,
            )
        )
        return

    print_text_report(manifest, plans, profile=profile)


if __name__ == "__main__":
    main()
